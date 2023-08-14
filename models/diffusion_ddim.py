
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import numpy as np
from datetime import datetime

# Loading modules
from models.diffusion_ddpm import Diffusion_DDPM
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from models.Unet_FiLmLayer import *
from models.simple_Unet import * 
from models.encoder.autoencoder import *
from utils.print_utils import *
from utils.plot_utils import *

class Diffusion_DDIM(Diffusion_DDPM):        


# ==================== Sampling ====================
    def sample(self, batch, option = None):
        """
         For generating samples, we only need observation data as input.
        Structure of batch:
        batch = {
                image = (B, obs_horizon  , 128)
                position = (B, obs_horizon  , 2)
                velocity = (B, obs_horizon  , 2)
                actions = (B, obs_horizon  , 3)
        }
        The batch includes only observation data.
        The first obs_horizon entries are used for conditioning the model (contextual input to the model)

        option: 'sample_history': Retain the history of all denoising steps
                'None': Only return the final denoised image
        """

        for key, tensor in batch.items():
            batch[key] = tensor.to(self.device)

        observation_batch = batch
        # Create Condition vectors for the model
        obs_cond = self.prepare_obs_cond_vectors(observation_batch) # (B, obs_horizon, obs_dim)
        obs_cond = obs_cond[0,...].unsqueeze(0).unsqueeze(1) # (, 1, obs_horizon, obs_dim)

        # Prepare an inpainting vector
        inpaint_vector  = self.prepare_inpaint_vectors(observation_batch) # (B, inpainting_horizon, pred_dim)
        inpaint_vector = inpaint_vector[0,...].unsqueeze(0).unsqueeze(1) # (1, 1, inpainting_horizon, pred_dim)
        B = obs_cond.shape[0]
        x_t = torch.rand(1, 1, self.pred_horizon + self.inpaint_horizon, self.prediction_dim, device=self.device)    

        if option == 'sample_history':
            # init scheduler and vector to store all samples    
            sampling_history = [x_t]
            self.noise_scheduler.set_timesteps(self.noise_steps)
            for i, t in enumerate(self.noise_scheduler.timesteps):
                with torch.no_grad():
                    est_noise = self.noise_estimator(x_t, torch.tensor([t], device=self.device), obs_cond)
                x_t = self.noise_scheduler.step(est_noise, t, x_t).prev_sample
                x_t = self.add_constraints(x_t, inpaint_vector)
                sampling_history.append(x_t)
            return sampling_history # Return a list of all successive samples x_T to x_0

        # init scheduler stepsize
        self.noise_scheduler.set_timesteps(self.noise_steps)
        for i, t in enumerate(self.noise_scheduler.timesteps):
            # 1. predict noise residual
            with torch.no_grad():
                est_noise = self.noise_estimator(x_t, torch.tensor([t], device=self.device), obs_cond)
            x_t = self.noise_scheduler.step(est_noise, t, x_t).prev_sample
            x_t = self.add_constraints(x_t, inpaint_vector)
        return x_t

    # def prepare_inpaint_vectors(self, observation_batch):
    #     """
    #     Extract inpaint horizon data from observation batch from the back
    #     Adjust depending on the desired model output 

    #     """
    #     inpaint_position_vector = observation_batch['position'][:,-self.inpaint_horizon:,:]
    #     # inpaint_action_vector = observation_batch['action'][:,-self.inpaint_horizon:,:]

    #     # return torch.cat([inpaint_position_vector, inpaint_action_vector], dim=-1) # concat along state dim
    #     return inpaint_position_vector

