import torch

class DiffusionModel:
    def __init__(self, start_schedule=1e-4, end_schedule=0.02, timesteps=300):
        self.start_schedule = start_schedule
        self.end_schedule = end_schedule
        self.timesteps = timesteps
        self.betas = torch.linspace(start_schedule, end_schedule, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def forward(self, x_0, t, device):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(
            self.alphas_cumprod.sqrt(), t, x_0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            torch.sqrt(1.0 - self.alphas_cumprod), t, x_0.shape
        )

        mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

        return mean + variance, noise.to(device)

    @torch.no_grad()
    def backward(self, x, t, model, **kwargs):
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            torch.sqrt(1.0 - self.alphas_cumprod), t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(
            torch.sqrt(1.0 / self.alphas), t, x.shape
        )
        mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, **kwargs) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = betas_t

        if t == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            variance = torch.sqrt(posterior_variance_t) * noise
            return mean + variance

    @staticmethod
    def get_index_from_list(values, t, x_shape):
        batch_size = t.shape[0]
        """
        pick the values from vals
        according to the indices stored in `t`
        """
        result = values.gather(-1, t.cpu())
        """
        if 
        x_shape = (5, 3, 64, 64)
            -> len(x_shape) = 4
            -> len(x_shape) - 1 = 3
            
        and thus we reshape `out` to dims
        (batch_size, 1, 1, 1)
        
        """
        return result.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)