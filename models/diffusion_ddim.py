
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
    def sample(self, batch,  step_size = 20, ddpm_steps = 100):
        # Prepare Data
        x_0, obs_cond = self.prepare_pred_cond_vectors(batch)
        x_0 = x_0[0, ...].unsqueeze(0).unsqueeze(1)
        obs_cond = obs_cond[0, ...].unsqueeze(0).unsqueeze(1)
        
        # Extract observations
        position_observation = obs_cond.squeeze()[:, :2].detach().cpu().numpy()
        actions_observation = obs_cond.squeeze()[:, 2:].detach().cpu().numpy()
        
        positions_groundtruth = x_0.squeeze()[:, :2].detach().cpu().numpy()
        actions_groundtruth = x_0.squeeze()[:, 2:].squeeze().detach().cpu().numpy()
        
        # Backward Process
        t_subset = torch.arange(0, self.noise_steps, step_size, device=self.device).long()
        x = torch.randn_like(x_0)
        t = t_subset[-1]
        sampling_history = [x.squeeze().detach().cpu().numpy()]

        for t_next in reversed(t_subset[:-1]):
            a_t = self.alphas_cumprod[t]
            a_tnext = self.alphas_cumprod[t_next]

            x_1 = a_tnext.sqrt() * (x - (1 - a_t).sqrt() * self.noise_estimator(x, t, obs_cond)) / a_t.sqrt()
            x_2 = (1 - a_tnext).sqrt() * self.noise_estimator(x, t, obs_cond)

            x = x_1 + x_2
            x = self.add_constraints(x, x_0)
            sampling_history.append(x.squeeze().detach().cpu().numpy().copy())
            t = t_next


        #         # Create a subet of all timesteps
        # timesteps = np.flip(np.arange(noise_steps)) # [0 1 2 3 4 5 6 7 8 9, ... , noise_steps]
        # t_subset = timesteps[::step_size] # [0 2 4 6 8, ... , noise_steps]

        # for idx, timestep in enumerate(t_subset):

        #     timestep_prev = t_subset[idx+1] if idx > len(t_subset) else None
        #     alpha_bar_t = self.alphas_cumprod[timestep]
        #     alpha_bar_t_prev = self.alphas_cumprod[timestep_prev] if timestep_prev is not None else 0


        #     x_1 = alpha_bar_t_prev.sqrt() * (x - (1 - alpha_bar_t).sqrt() * self.noise_estimator(x, t, obs_cond)) / alpha_bar_t.sqrt()
        #     x_2 = (1 - alpha_bar_t_prev).sqrt() * self.noise_estimator(x, timestep, obs_cond)

        #     x = x_1 + x_2
        #     x = self.add_constraints(x, inpaint_vector)
        # return x
        
        for t in reversed(range(0, ddpm_steps)):
            x = self.p_reverseProcess(obs_cond, x, t)
            x = self.add_constraints(x, x_0)
            sampling_history.append(x.squeeze().detach().cpu().numpy().copy())
            
        print("Sampling finished.")
        print("Sampling history length: ", len(sampling_history))
    
        plt_toVideo(self,
                sampling_history,
                positions_groundtruth = positions_groundtruth,
                position_observation = position_observation,
                actions_groundtruth = actions_groundtruth,
                actions_observation = actions_observation)

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




