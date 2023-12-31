import math
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, label):
        super().__init__()

        if label:
            self.label_embedding = nn.Embedding(1, 8)
            self.label_mlp = nn.Linear(1, out_channels)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(0.03)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            self.dropout,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t, label=None):
        h = self.conv1(x)
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        if label:
            label_emb = self.relu(self.label_mlp(label))
            label_emb = label_emb[(...,) + (None,) * 2]
            h = h + label_emb
        # Second Conv
        h = self.conv2(h)
        return h


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNet(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        channels=(64, 128, 256, 512),
        time_emb_dim=32,
        label=None
    ):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsample
        self.downs = nn.ModuleList()
        for channel in channels:
            self.downs.append(Block(input_channels, channel, time_emb_dim, label))
            input_channels = channel

        # Bottleneck
        self.bottleneck = Block(channels[-1], channels[-1] * 2, time_emb_dim, label)

        # Upsample
        self.ups = nn.ModuleList()
        for channel in reversed(channels):
            self.ups.append(
                nn.ConvTranspose2d(channel * 2, channel, kernel_size=2, stride=2)
            )
            self.ups.append(Block(channel * 2, channel, time_emb_dim, label))

        self.output = nn.Conv2d(channels[0], output_channels, kernel_size=1)
        self.dropouts = [down.dropout for down in self.downs] + [up.dropout for up in self.ups if 'dropout' in up.__dict__] + [self.bottleneck.dropout]

    def forward(self, x, timestep, label=None):
        # Embedd time
        t = self.time_mlp(timestep)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t, label)
            residual_inputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, t)
        for i in range(0, len(self.ups), 2):
            conv_t = self.ups[i]
            up = self.ups[i + 1]
            residual_x = residual_inputs.pop()

            x = conv_t(x)

            if x.shape != residual_x.shape:
                x = TF.resize(x, size=residual_x.shape[2:], antialias=True)

            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t, label)

        return self.output(x)