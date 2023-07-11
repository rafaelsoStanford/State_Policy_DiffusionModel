
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import numpy as np
from datetime import datetime

# Loading modules
from models.Unet_FiLmLayer import *
from models.simple_Unet import * 
from models.encoder.autoencoder import *
from utils.schedulers import *
from utils.print_utils import *
from utils.plot_utils import *
from utils.data_utils import *


class Diffusion_DDPM(pl.LightningModule):
    def __init__(self
                , noise_steps=1000
                , denoising_steps=1000
                , obs_horizon = 10
                , pred_horizon= 10
                , observation_dim = 2
                , prediction_dim = 2
                , learning_rate = 1e-4
                , model = 'UNet'
                , vision_encoder = None
                , noise_scheduler = 'linear'
                , inpaint_horizon = 10
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.date = datetime.today().strftime('%Y_%m_%d_%H-%M-%S')
# ==================== Init ====================
    # --------------------- Diffusion params ---------------------
        self.noise_steps = self.hparams.noise_steps
        self.denoising_steps = self.hparams.denoising_steps
        self.NoiseScheduler = None
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.observation_dim = observation_dim
        self.prediction_dim = prediction_dim
        self.inpaint_horizon = inpaint_horizon

    # --------------------- Model Architecture ---------------------
        if model == 'UNet_Film':
            print("Loading UNet with FiLm conditioning")
            self.model = UNet_Film
        else:
            print("Loading UNet (simple) ")
            self.model = UNet

    # --------------------- Noise Schedule Params---------------------
        if noise_scheduler == 'linear_v2':
            self.NoiseScheduler = linear_beta_schedule_v2
        if noise_scheduler == 'linear':
            self.NoiseScheduler = linear_beta_schedule
        if noise_scheduler == 'cosine_beta_schedule':
            self.NoiseScheduler = cosine_beta_schedule

        betas =  self.NoiseScheduler(self, noise_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    # --------------------- Model --------------------- 
        self.lr = learning_rate  
        self.loss = nn.MSELoss()
        self.noise_estimator = self.model(
                                    in_channels= 1,
                                    out_channels= 1,
                                    noise_steps= noise_steps,
                                    global_cond_dim= (observation_dim) * obs_horizon, # 512 is the output dim of Resnet18, 2 is the position dim
                                    time_dim = 256 # Embedding dimension for time (t) of the current denoising step
                                )

        ### Define model which will be a simplifed 1D UNet
        if vision_encoder == 'resnet18':
            print("Loading Resnet18")
            self.vision_encoder = VisionEncoder() # Loads pretrained weights of Resnet18 with output dim 512 (also modified layers as Suggested by Song et al.)
        
        else:
            print("Loading lightweight Autoencoder")
            vision = autoencoder.load_from_checkpoint(checkpoint_path="./tb_logs_autoencoder/version_23/checkpoints/epoch=25.ckpt")
            self.vision_encoder = vision.encoder
        self.vision_encoder.device = self.device
        self.vision_encoder.eval() # 128 entries

        # --------------------- Output environment settings ---------------------
        if os.getenv("LOCAL_RANK", '0') == '0':
            print_hyperparameters(
            obs_horizon, pred_horizon, observation_dim, prediction_dim, noise_steps, inpaint_horizon, model, learning_rate, vision_encoder)
            # print("Model Architecture: ", self.noise_estimator)

# ==================== Training ====================
    def training_step(self, batch, batch_idx):
        loss = self.onepass(batch, batch_idx, mode="train")
        self.log("train_loss",loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss

# ==================== Testing ====================
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.sample(batch, mode="test", step_size = 20 )

# ==================== Validation ====================    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.sample(batch, mode="validation")
        loss = self.onepass(batch, batch_idx, mode="validation")
        self.log("val_loss",loss,  sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5) # patience in the unit of epoch
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }

# ==================== Noising / Denoising Processes ====================
    def onepass(self, batch, batch_idx, mode="train"):
        # ---------------- Preparing Observation / Prediction data ----------------
        x_0 , obs_cond = self.prepare_pred_cond_vectors(batch)
        x_0 = x_0.unsqueeze(1)
        obs_cond = obs_cond.unsqueeze(1)
        B = x_0.shape[0]

        # ---------------- Forward Process ----------------
        t = torch.randint(0, self.noise_steps, (B,), device=self.device).long() # Values from [0, 999]
        noise = torch.randn_like(x_0)
        x_noisy = self.q_forwardProcess(x_0, t, noise) # (B, 1 , pred_horizon, pred_dim)
        x_noisy = self.add_constraints(x_noisy, x_0)

        # ---------------- Estimate noise / Single Backward process ----------------
        # Estimate noise using noise_predictor
        if mode == "train":
            noise_estimated = self.noise_estimator(x_noisy, t, obs_cond)
        else:
            with torch.no_grad():
                noise_estimated = self.noise_estimator(x_noisy, t, obs_cond)

        # ----------------  Loss ----------------
        loss = self.loss(noise, noise_estimated) #MSE Loss
        return loss
    
# ==================== Sampling ====================
    def sample(self, batch, mode):
        # ---------------- Prepare Data ----------------
        x_0 , obs_cond = self.prepare_pred_cond_vectors(batch)
        x_0 = x_0[0,...].unsqueeze(0).unsqueeze(1)
        obs_cond = obs_cond[0,...].unsqueeze(0).unsqueeze(1)
        # Observations ie Past
        position_observation = obs_cond.squeeze()[:, :2].detach().cpu().numpy() 
        actions_observation = obs_cond.squeeze()[:, 2:].detach().cpu().numpy()

        positions_groundtruth = x_0.squeeze()[:, :2].detach().cpu().numpy() #(20 , 2)
        actions_groundtruth = x_0.squeeze()[:, 2:].squeeze().detach().cpu().numpy() #(20 , 3)
    
        # ---------------- Plotting ----------------
        if mode == 'validation':
            x_0_predicted = self.p_reverseProcess_loop(x_cond = obs_cond, x_0 = x_0) # Only one batch is used to be visualized

            # Predictions
            positions_predicted = x_0_predicted.squeeze()[:, :2].detach().cpu().numpy() #(20 , 2)
            actions_predicted = x_0_predicted.squeeze()[:, 2:].detach().cpu().numpy() #(20 , 3)

            # Plot to tensorboard
            plt_toTensorboard(self,
                positions_predicted = positions_predicted,
                positions_groundtruth = positions_groundtruth,
                position_observation = position_observation,
                actions_predicted = actions_predicted,
                actions_groundtruth = actions_groundtruth,
                actions_observation = actions_observation,
            )
            
        if mode == 'test':
            # Sample and save to video
            sampling_history = []
            x_t = torch.rand(1, 1, self.pred_horizon + self.inpaint_horizon, self.prediction_dim, device=self.device)

            for t in reversed(range(0,self.denoising_steps)): # t ranges from denoising_steps-1 to 0
                x_t =  self.p_reverseProcess(obs_cond,  x_t,  t)
                x_t = self.add_constraints(x_t, x_0)
                sampling_history.append(x_t.squeeze().detach().cpu().numpy())
            
            plt_toVideo(self,
                sampling_history,
                positions_groundtruth = positions_groundtruth,
                position_observation = position_observation,
                actions_groundtruth = actions_groundtruth,
                actions_observation = actions_observation)  
            
            return sampling_history[-1] # (1, 1, pred_horizon, 5)
            
        

         # q(x_t | x_0)
    def q_forwardProcess(self, x_start, t, noise):
        x_t = torch.sqrt(self.alphas_cumprod[t])[:,None,None,None] * x_start + torch.sqrt(1-self.alphas_cumprod[t])[:,None,None,None] * noise
        return x_t

    @torch.no_grad()
    def p_reverseProcess_loop(self, x_cond, x_0 , x_T = None):
        if x_T is None:
            x_t = torch.rand(1, 1, self.pred_horizon + self.inpaint_horizon, self.prediction_dim, device=self.device) + x_0[:, : , self.inpaint_horizon, :]
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

    def add_constraints(self, x_t , x_0):
        x_t[:, : , :self.inpaint_horizon, :] = x_0[:, : , :self.inpaint_horizon, :].clone() # inpaint datapoints 
        return x_t

    # ==================== Helper functions ====================
    def prepare_pred_cond_vectors(self, batch):
        
        normalized_img                                  =  batch['image'][:,:self.obs_horizon ,:].to(self.device) 
        normalized_pos                                  =  batch['position'][:,:self.obs_horizon ,:].to(self.device) 
        normalized_act                                  =  batch['action'][:,:self.obs_horizon,:].to(self.device) 
        normalized_vel                                  =  batch['velocity'][:,:self.obs_horizon ,:].to(self.device) 

        # ---------------- Encoding Image data ----------------
        encoded_img = self.vision_encoder(normalized_img.flatten(end_dim=1)) # (B, 128)
        image_features = encoded_img.reshape(*normalized_img.shape[:2],-1) # (B, t_0:t_obs , 128)

        # ---------------- Conditional vector ----------------
        # Concatenate position and action data and image features
        obs_cond = torch.cat([normalized_pos, normalized_act,normalized_vel, image_features], dim=-1) # (B, t_0:t_obs, 512 + 3 + 2)

        # ---------------- Preparing Prediction data (acts as ground truth) ----------------
        x_0_pos = batch['position'][:,self.obs_horizon: ,:].to(self.device) # (B, t_obs:t_pred , 2)
        #x_0_pos = (x_0_pos - translation_vec[:, None, :]) / 2.0 # Normalizing to the same frame as the observation data
        x_0_act = batch['action'][:, self.obs_horizon: ,:].to(self.device) # (B, t_obs:t_pred, 3)
        x_0 = torch.cat([x_0_pos, x_0_act], dim=-1) # (B, t_obs:t_pred, 5)

        # Adding past obervation as inpainting condition
        x_0 = torch.cat((obs_cond[:, -self.inpaint_horizon:, :5], x_0) , dim=1) # Concat in time dim

        # ---------------- Assert cond dimensions compatible with model (important when preloading / changing conditioning data) ----------------
        assert(obs_cond.shape[-1]*self.obs_horizon == self.noise_estimator.down1.cond_encoder[2].state_dict()['weight'].shape[1]) # Check if cond dim is correct
        return x_0 , obs_cond
