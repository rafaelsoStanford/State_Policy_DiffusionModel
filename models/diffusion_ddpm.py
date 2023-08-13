
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import numpy as np
from datetime import datetime
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Loading modules
from models.Unet_FiLmLayer import *
from models.simple_Unet import * 
from models.Unet_FiLmLayer_noAttention import *
from models.encoder.autoencoder import *
from utils.schedulers import *
from utils.print_utils import *
from utils.plot_utils import *



class Diffusion_DDPM(pl.LightningModule):
    def __init__(self
                , noise_steps=1000
                , obs_horizon = 10
                , pred_horizon= 10
                , observation_dim = 2
                , prediction_dim = 2
                , learning_rate = 1e-4
                , model = 'UNet'
                , vision_encoder = None
                , noise_scheduler = 'linear'
                , inpaint_horizon = 10
                , step_size = 1
                ):
        super().__init__()

        self.save_hyperparameters()
        self.date = datetime.today().strftime('%Y_%m_%d_%H-%M-%S')
# ==================== Init ====================
    # --------------------- Diffusion params ---------------------
        self.noise_steps = self.hparams.noise_steps
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
        elif model == 'UNet_FilmnoAttention':
            print("Loading UNet with FiLm conditioning, no attention Layers")
            self.model = UNet_Film_noAttention
        else:
            print("Loading UNet (simple) ")
            self.model = UNet

    # # --------------------- Noise Schedule Params---------------------
        if noise_scheduler == 'linear_v2':
            self.NoiseScheduler = linear_beta_schedule_v2
        if noise_scheduler == 'linear':
            self.NoiseScheduler = linear_beta_schedule
        if noise_scheduler == 'cosine_beta_schedule':
            self.NoiseScheduler = cosine_beta_schedule

#! Scheduler to be exchanges with DDPMScheduler (from diffusers.schedulers.scheduling_ddpm import DDPMScheduler)
        # self.noise_scheduler = DDPMScheduler(
        #     num_train_timesteps=self.noise_steps,
        #     beta_schedule='squaredcos_cap_v2',
        #     clip_sample=True,
        #     prediction_type='epsilon'
        # )


        # ? Not needed if DDPMScheduler works
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
        # Model parameters
        self.lr = learning_rate  
        self.loss = nn.MSELoss()
        self.noise_estimator = self.model(
                                    in_channels= 1,
                                    out_channels= 1,
                                    noise_steps= noise_steps,
                                    global_cond_dim= (observation_dim) * obs_horizon, # 512 is the output dim of Resnet18, 2 is the position dim
                                    time_dim = 256 # Embedding dimension for time (t) of the current denoising step
                                )

        print("Loading lightweight Autoencoder")
        vision = autoencoder.load_from_checkpoint(checkpoint_path="./tb_logs_autoencoder/version_23/checkpoints/epoch=25.ckpt")
        self.vision_encoder = vision.encoder
        self.vision_encoder.device = self.device
        self.vision_encoder.eval() # 128 entries


# ==================== Training ====================
    def training_step(self, batch, batch_idx):
        loss = self.process_single_batch(batch)
        self.log("train_loss",loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss


# ==================== Validation ====================    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            x_0_predicted , observation_batch , inpaint_vector = self.validate(batch)
            # Plot to tensorboard
            self.plt2tensorboard(
                batch = batch,
                prediction=x_0_predicted,
                inpaint_vector=inpaint_vector,
                observation_batch=observation_batch,
            )
            
        loss = self.process_single_batch(batch)
        self.log("val_loss",loss,  sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }

# ==================== Noising / Denoising Processes ====================
    def process_single_batch(self, batch):
        """
            Structure of batch:
            batch = {
                image = (B, 1, obs_horizon + pred_horizon, 128)
                position = (B, 1, obs_horizon + pred_horizon, 2)
                velocity = (B, 1, obs_horizon + pred_horizon, 2)
                actions = (B, 1, obs_horizon + pred_horizon, 3)
            }

            The first obs_horizon entries are used for conditioning the model (contextual input to the model)
            The last pred_horizon entries are needed for the forward process vector 
        """

        # ---------------- Preparing Observation / Prediction data ----------------
        # Sepearate observation and prediction data
        observation_batch = self.prepare_observation_batch(batch)
        prediction_batch = self.prepare_prediction_batch(batch)
        
        # Create Condition vectors for the model
        obs_cond = self.prepare_obs_cond_vectors(observation_batch) # (B, obs_horizon, obs_dim)
        obs_cond = obs_cond.unsqueeze(1) # (B, 1, obs_horizon, obs_dim)
        # print("obs_cond shape: ", obs_cond.shape)
        
        # Prepare prediction data vector for the forward process
        x_0 = self.prepare_prediction_vectors(prediction_batch) # (B, pred_horizon, pred_dim)
        x_0 = x_0.unsqueeze(1) # (B, 1, pred_horizon, pred_dim)
        # print("x_0 shape: ", x_0.shape)

        # Prepare an inpainting vector
        x_0_inpaint = self.prepare_inpaint_vectors(observation_batch) # (B, inpainting_horizon, pred_dim)
        x_0_inpaint = x_0_inpaint.unsqueeze(1) # (B, 1, inpainting_horizon, pred_dim)
        # print("x_0_inpaint shape: ", x_0_inpaint.shape)
        B = x_0.shape[0]


        # ---------------- Forward Process ----------------
        # Sample noise steps
        t = torch.randint(0, self.noise_steps, (B,), device=self.device).long() # Value range [0, 999]
        # Prepare  prediction vector:
        prediction_vector = torch.cat( [x_0_inpaint, x_0] , dim=2) # Concat in time dim
        noise = torch.randn_like(prediction_vector)


        x_noisy = self.q_forwardProcess(prediction_vector, t, noise) # (B, 1 , pred_horizon, pred_dim)
        x_noisy = self.add_constraints(x_noisy, x_0_inpaint)

        # ---------------- Estimate noise / Single Backward process ----------------
        noise_estimated = self.noise_estimator(x_noisy, t, obs_cond)

        # ----------------  Loss ----------------
        loss = self.loss(noise, noise_estimated) #MSE Loss
        return loss
    
        # q(x_t | x_0)
    def q_forwardProcess(self, x_start, t, noise):
        x_t = torch.sqrt(self.alphas_cumprod[t])[:,None,None,None] \
                * x_start + torch.sqrt(1-self.alphas_cumprod[t])[:,None,None,None] * noise
        return x_t
    
# ==================== Validation ====================
    def validate(self, batch):
        """
            For generating samples, we only need observation data as input.
            This function is meant to be used for validation during training.


            Structure of batch:
            batch = {
                image = (B, obs_horizon + pred_horizon, 128)
                position = (B, obs_horizon + pred_horizon, 2)
                velocity = (B, obs_horizon + pred_horizon, 2)
                actions = (B, obs_horizon + pred_horizon, 3)
            }
            The batch includes both observation and prediction data.
            The first obs_horizon entries are used for conditioning the model (contextual input to the model)
        """
        # Sepearate observation and prediction data
        observation_batch = self.prepare_observation_batch(batch)

        # Create Condition vectors for the model
        obs_cond = self.prepare_obs_cond_vectors(observation_batch) # (B, obs_horizon, obs_dim)
        obs_cond = obs_cond[0,...].unsqueeze(0).unsqueeze(1) # (, 1, obs_horizon, obs_dim)

        # Prepare an inpainting vector
        inpaint_vector  = self.prepare_inpaint_vectors(observation_batch) # (B, inpainting_horizon, pred_dim)
        inpaint_vector = inpaint_vector[0,...].unsqueeze(0).unsqueeze(1) # (1, 1, inpainting_horizon, pred_dim)
        B = obs_cond.shape[0]


        x_0_predicted = self.p_reverseProcess_loop(x_cond = obs_cond, x_inpaint = inpaint_vector)
        return x_0_predicted , observation_batch , inpaint_vector
            
    # p(x_t-1 | x_t)
    @torch.no_grad()
    def p_reverseProcess_loop(self, x_cond, x_inpaint, x_T = None):
        """
        Conditioned reverse process loop from x_T to x_0, where you inpaint the last inpaint_horizon entries of x_t

        Output shape: (B, 1, inpaint_horizon + pred_horizon, pred_dim)

        """
        if x_T is None:
            x_t = torch.rand(1, 1, self.pred_horizon + self.inpaint_horizon, self.prediction_dim, device=self.device)
        else:
            x_t = x_T
        
        for t in reversed(range(0,self.noise_steps)): # t ranges from 999 to 0
            if t == 0:
                z = torch.zeros_like(x_t) 
            else:
                z = torch.randn_like(x_t)

            est_noise = self.noise_estimator(x_t, torch.tensor([t], device=self.device), x_cond)

            x_t = self.add_constraints(x_t, x_inpaint)            
            x_t = 1/torch.sqrt(self.alphas[t]) * (x_t-(1-self.alphas[t])/torch.sqrt(1-self.alphas_cumprod[t])*est_noise)  +  torch.sqrt(self.betas[t])*z
            x_t = self.add_constraints(x_t, x_inpaint)
        
        return x_t # (1, 1, pred_horizon + inpaint_horizon, pred_dim)


    def add_constraints(self, x_t , x_inpaint):
        # Reverse x_inpaint and overwrite the first inpaint_horizon entries of x_t
        x_t[:, : , :self.inpaint_horizon, :] = x_inpaint
        return x_t


    # ==================== Sampling ====================
    def sample(self, batch):
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
        """

        # Sepearate observation and prediction data
        observation_batch = batch

        # Create Condition vectors for the model
        obs_cond = self.prepare_obs_cond_vectors(observation_batch) # (B, obs_horizon, obs_dim)
        obs_cond = obs_cond[0,...].unsqueeze(0).unsqueeze(1) # (, 1, obs_horizon, obs_dim)

        # Prepare an inpainting vector
        inpaint_vector  = self.prepare_inpaint_vectors(observation_batch) # (B, inpainting_horizon, pred_dim)
        inpaint_vector = inpaint_vector[0,...].unsqueeze(0).unsqueeze(1) # (1, 1, inpainting_horizon, pred_dim)
        B = obs_cond.shape[0]

        x_0_predicted = self.p_reverseProcess_loop(x_cond = obs_cond, x_inpaint = inpaint_vector)
        return x_0_predicted # (1, 1, inpaint_horizon + pred_horizon, pred_dim)


    # ==================== Helper functions ====================
    def prepare_observation_batch(self, batch):
        """
        Prepares the observation batch for the model
        """
        normalized_img    =  batch['image'][:,:self.obs_horizon ,:].to(self.device) 
        normalized_pos    =  batch['position'][:,:self.obs_horizon ,:].to(self.device) 
        normalized_act    =  batch['action'][:,:self.obs_horizon,:].to(self.device) 
        normalized_vel    =  batch['velocity'][:,:self.obs_horizon ,:].to(self.device) 

        observation_batch = {
            'image': normalized_img,
            'position': normalized_pos,
            'action': normalized_act,
            'velocity': normalized_vel
        }
        return observation_batch
    
    def prepare_prediction_batch(self, batch):
        """
        Prepares the prediction batch for the model
        """
        normalized_img    =  batch['image'][:,self.obs_horizon: ,:].to(self.device) 
        normalized_pos    =  batch['position'][:,self.obs_horizon: ,:].to(self.device) 
        normalized_act    =  batch['action'][:,self.obs_horizon: ,:].to(self.device) 
        normalized_vel    =  batch['velocity'][:,self.obs_horizon: ,:].to(self.device) 

        prediction_batch = {
            'image': normalized_img,
            'position': normalized_pos,
            'action': normalized_act,
            'velocity': normalized_vel
        }
        return prediction_batch
    
    def prepare_obs_cond_vectors(self, observation_batch):
        img_data = observation_batch['image']
        # ---------------- Encoding Image data ----------------
        encoded_img = self.vision_encoder(img_data.flatten(end_dim=1)) # (B, 128)
        image_features = encoded_img.reshape(*img_data.shape[:2],-1) # (B, t_0:t_obs , 128)

        # ---------------- Conditional vector ----------------
        # Concatenate position and action data and image features
        obs_cond = torch.cat([  observation_batch['position'], 
                                observation_batch['action'],
                                observation_batch['velocity'], 
                                image_features], 
                                dim=-1)        # (B, t_0:t_obs , 128+4+2)
        return obs_cond

    def prepare_prediction_vectors(self, prediction_batch):
        # ---------------- Preparing Prediction data  ----------------
        # Concatenate position and action data
        position_data = prediction_batch['position']
        action_data = prediction_batch['action']
        x_0 = torch.cat([position_data, action_data], dim=-1) # (B, t_obs:t_pred , 3+2)
        return x_0
    
    def prepare_inpaint_vectors(self, observation_batch):
        """
        Extract inpaint horizon data from observation batch from the back

        """
        inpaint_position_vector = observation_batch['position'][:,-self.inpaint_horizon:,:]
        inpaint_action_vector = observation_batch['action'][:,-self.inpaint_horizon:,:]

        return torch.cat([inpaint_position_vector, inpaint_action_vector], dim=-1) # concat along state dim
        

    def plt2tensorboard(self , batch , prediction , inpaint_vector , observation_batch):
        
        # Data from observation batch
        position_observation = observation_batch['position'].cpu().numpy() # (B, t_obs, 2)
        position_observation = position_observation[0] # (t_obs, 2)

        positions_inpainted = inpaint_vector.squeeze().cpu().numpy() # (t_inpaint, 5)
        positions_inpainted = positions_inpainted[:, :2] # (t_inpaint, 2)

        positions_groundtruth = batch['position'].cpu().numpy() # (B, t_pred, 2)
        positions_groundtruth = positions_groundtruth[0] # (t_pred, 2)

        positions_predicted = prediction.squeeze().cpu().numpy() # (1, t_pred, 5) # single batch was predicted
        positions_predicted = positions_predicted[:, :2] # (t_pred, 2)

        # ---------------- Plotting ----------------
        writer = self.logger.experiment
        niter  = self.global_step
        plt.switch_backend('agg')
        # ---------------- 2D Position Plot ----------------
        fig = plt.figure()
        fig.clf()
        # Create a colormap for fading colors based on the number of timesteps
        cmap = get_cmap('viridis', self.pred_horizon + self.inpaint_horizon)
        # Create an array of indices from 0 to timesteps-1
        indices = np.arange(self.pred_horizon + self.inpaint_horizon)
        # Normalize the indices to the range [0, 1]
        normalized_indices = indices / (self.pred_horizon + self.inpaint_horizon - 1)
        # Create a color array using the colormap and normalized indices
        colors = cmap(normalized_indices)

        # Plotting with labels for legend
        plt.plot(positions_groundtruth[:,0], positions_groundtruth[:,1],'g.', label="Ground Truth")
        plt.plot(position_observation[:, 0], position_observation[:,1],'b.', label="Observation")

        plt.scatter(positions_predicted[:,0],positions_predicted[:,1],color=colors, s=10, label="Predicted Positions")
        plt.scatter(positions_inpainted[:,0],positions_inpainted[:,1],color='r', s=20, label="Inpainted Positions")

        # Adding labels to the x and y axes
        plt.xlabel('X Position')
        plt.ylabel('Y Position')

        # Displaying the legend
        plt.legend(loc="upper right")

        plt.grid()
        plt.axis('equal')

        # Plot to tensorboard
        plt2tsb(fig, writer, 'Predicted_path ' + self.date , niter)
        plt.close("all")

        
