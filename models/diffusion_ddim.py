
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import numpy as np
from datetime import datetime

# Loading modules
from models.diffusion_ddpm import Diffusion_DDPM
from models.Unet_FiLmLayer import *
from models.simple_Unet import * 
from models.encoder.autoencoder import *
from utils.print_utils import *
from utils.plot_utils import *

class Diffusion_DDIM(Diffusion_DDPM):
    """
    Only difference between DDPM and DDIM is the generation process, the previously trained noise estimator can still be used.
    Thus we only overwrite the generation process (sample function).
    """
# ==================== Sampling ====================
    def sample(self, batch,  step_size = 20, ddpm_steps = None):
        
        """
        Generate a sample from the model using DDIM. Training is not required, as a trained noise estimator from DDPM can be used, without any modification.

        Parameters
        ----------
        batch : torch.Tensor
            Batch of observations.
        step_size : int, optional
            Number of steps between each sample. The default is 20.
        ddpm_steps : int, optional  
            Number of steps to run the DDPM process. The default is None. If not None, the DDIM process is run first, then the DDPM process.
            This can be considered a hybrid between DDIM and DDPM.

        Returns:
        --------
        sampling_history : list
            List of all denoising steps of the Markov chain.
        """
        
        x_0, obs_cond = self.prepare_pred_cond_vectors(batch)
        x_0 = x_0[0, ...].unsqueeze(0).unsqueeze(1)
        obs_cond = obs_cond[0, ...].unsqueeze(0).unsqueeze(1)

        # Sampling
        t_subset = torch.arange(0, self.noise_steps, step_size, device=self.device).long()
        x = torch.randn_like(x_0)
        t = t_subset[-1]
        sampling_history = [x.squeeze().detach().cpu().numpy()]
        for t_next in reversed(t_subset[:-1]):
            a_t = self.alphas_cumprod[t]
            a_tnext = self.alphas_cumprod[t_next]

            x_1 = a_tnext.sqrt() * (x - (1 - a_t).sqrt() * self.noise_estimator(x, t, obs_cond)) / a_t.sqrt()
            x_2 = (1 - a_tnext).sqrt() * self.noise_estimator(x, t_next, obs_cond)

            x = x_1 + x_2
            x = self.add_constraints(x, x_0)
            sampling_history.append(x.squeeze().detach().cpu().numpy().copy())
            t = t_next
        
        if ddpm_steps is not None:
            for t in reversed(range(0, ddpm_steps)):
                x = self.p_reverseProcess(obs_cond, x, t)
                x = self.add_constraints(x, x_0)
                sampling_history.append(x.squeeze().detach().cpu().numpy().copy())
        return sampling_history

    # q(x_t | x_0)
    def q_forwardProcess(self, x_start, t, noise):
        x_t = torch.sqrt(self.alphas_cumprod[t])[:,None,None,None] * x_start + torch.sqrt(1-self.alphas_cumprod[t])[:,None,None,None] * noise
        return x_t

    @torch.no_grad()
    def p_reverseProcess_loop(self, x_cond, x_0 , x_T = None):
        if x_T is None:
            x_t = torch.rand(1, 1, self.pred_horizon + self.inpaint_horizon, self.prediction_dim, device=self.device)
        else:
            x_t = x_T
        
        for t in reversed(range(0,self.noise_steps)): # t ranges from 999 to 0
            x_t =  self.p_reverseProcess(x_cond,  x_t,  t)

            x_t = self.add_constraints(x_t, x_0)

        return x_t

    @torch.no_grad()
    def p_reverseProcess(self, x_cond, x_t, t):
        if t == 0:
            z = torch.zeros_like(x_t)
        else:
            z = torch.randn_like(x_t)
        est_noise = self.noise_estimator(x_t, torch.tensor([t], device=self.device), x_cond)
        x_t = 1/torch.sqrt(self.alphas[t])* (x_t-(1-self.alphas[t])/torch.sqrt(1-self.alphas_cumprod[t])*est_noise) +  torch.sqrt(self.betas[t])*z
        return x_t




