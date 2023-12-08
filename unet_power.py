import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        #FIXME: different from the original paper UNet, didn't use maxpooling in this code
        #FIXME: potential explanation: cifar10 is too small, use conv to down sample 
        #FIXME: can keep more information and also enlarge the model capacity.
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x
    
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h

class ResBlock(nn.Module):  # resblock with time embedding
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None] 
        # directly sum up the latent feature and the time embedding
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h
    
class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout, in_ch=1):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'

        self.ood_detection_indicator = False
        self.record_latent = False

        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.head = nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch] # record output channel when downsample for upsample
        now_ch = ch

        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(now_ch, out_ch, tdim, 
                                                dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)
        
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(chs.pop()+now_ch, out_ch, tdim,
                                              dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0
    
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1),
        )
        self.initialize()
    
    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def _record_latent_features(self, record_timesteps):
        # latent recording
        self.record_latent = True
        self.record_timesteps = record_timesteps
        if self.record_latent:
            self.record_latent_features = { key:[] for key in record_timesteps }

    def _stop_record_latent_features(self):
        self.record_latent = False

    def _start_ood_detection(self, ood_detector, detect_timesteps):
        self.detect_timesteps = detect_timesteps
        self.ood_detection_indicator = True
        self.ood_detector = ood_detector
        self.ood_detect_res = []

    def _stop_ood_detection(self):
        self.ood_detection_indicator = False
    
    def forward(self, x, t):
        # timestep embedding
        temb = self.time_embedding(t)
        # downsampling
        h = self.head(x)
        # print(h.shape)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            # print(h.shape)
            hs.append(h)
        # middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        
        # recording 
        if self.record_latent:
            for i, _t in enumerate(t):
                if _t in self.record_timesteps:
                    self.record_latent_features[_t.item()].append(h[i].detach().cpu().numpy())
        
        if self.ood_detection_indicator:
            for i, _t in enumerate(t):
                if _t in self.detect_timesteps:
                    # print(h[i].detach().cpu().numpy().reshape(1,-1).shape)
                    # print(h[i].detach().cpu().numpy().reshape(1,-1))
                    ood_pred, max_dist = self.ood_detector.detect_l2_distance_ood(h[i].detach().cpu().numpy().reshape(1,-1), _t.item())
                    self.ood_detect_res.append((ood_pred, max_dist))
        
        # upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                # print(h.shape)
            h = layer(h, temb)
            # print(h.shape)
        h = self.tail(h)
        # print(h.shape)

        assert len(hs) == 0
        return h

if __name__ == '__main__':
    batch_size = 8
    model = UNet(T=1000, ch=32, ch_mult=[1,2,2,2], attn=[1], num_res_blocks=2, 
                 dropout=0.1)
    x = torch.randn(batch_size, 1, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)
                
