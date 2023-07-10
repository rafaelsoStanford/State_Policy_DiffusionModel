import torch
import numpy as np


# ==================== Schedulers ====================
def linear_beta_schedule(self, steps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    beta = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32, device=self.device)
    return beta


def linear_beta_schedule_v2(self, steps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 500 / steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    beta = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32, device=self.device)
    return beta


def cosine_beta_schedule(self, timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    betas =  torch.tensor(betas_clipped, dtype=dtype)
    return betas
