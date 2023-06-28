import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn
from PIL import Image
import io
from datetime import datetime
import pytorch_lightning as pl

# Loading modules
from models.Unet_FiLmLayer import *
from models.simple_Unet import * 
from models.encoder.autoencoder import *



class Diffusion(pl.LightningModule):
    def __init__(self, noise_steps=1000
                , obs_horizon = 10
                , pred_horizon= 10
                , observation_dim = 2
                , prediction_dim = 2
                , learning_rate = 1e-4
                , model = 'UNet'
                , vision_encoder = None
                , inpaint_horizon = 10):
        super().__init__()

        self.save_hyperparameters()
        self.date = datetime.today().strftime('%Y-%m-%d-%H')
    # ==================== Init ====================
    # --------------------- Diffusion params ---------------------
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.observation_dim = observation_dim
        self.prediction_dim = prediction_dim
        self.noise_steps = noise_steps
        self.NoiseScheduler = linear_beta_schedule

        self.inpaint_horizon = inpaint_horizon
    # --------------------- Model Architecture ---------------------
        if model == 'UNet_Film':
            print("Loading UNet with FiLm conditioning")
            self.model = UNet_Film
        else:
            print("Loading UNet (simple) ")
            self.model = UNet

    # --------------------- Noise Schedule Params---------------------
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
        
        print_hyperparameters(
           obs_horizon, pred_horizon, observation_dim, prediction_dim, noise_steps, inpaint_horizon, model, learning_rate, vision_encoder)

# ==================== Training ====================
    def training_step(self, batch, batch_idx):
        loss = self.onepass(batch, batch_idx, mode="train")
        self.log("train_loss",loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss

# ==================== Testing ====================
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.sample(batch, batch_idx, mode="test")

# ==================== Validation ====================    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.sample(batch, batch_idx, mode="validation")
        loss = self.onepass(batch, batch_idx, mode="validation")
        self.log("val_loss",loss)
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

        # # Inpaint: replace the first datapoint with the condition
        # x_noisy[:, : , :self.inpaint_horizon, :] = x_0[:, : , :self.inpaint_horizon, :].clone() # inpaint the first datapoint (should be enough)
        # x_noisy[:, :, :, 2] = torch.clip(x_noisy[:, :, :, 2].clone(), min=-1.0, max=1.0) # Enforce action limits (steering angle)
        # x_noisy[:, :, :, 3:] = torch.clip(x_noisy[:, :, :, 3:].clone(), min=0.0, max=1.0)   # Enforce action limits (acceleration and brake)

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
    


    # ---------------------- Sampling --------------------------------
    def sample(self, batch, batch_idx, mode):
    
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
            self.plt_toTensorboard(
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

            for t in reversed(range(0,self.noise_steps)): # t ranges from 999 to 0
                x_t =  self.p_reverseProcess(obs_cond,  x_t,  t)

                x_t = self.add_constraints(x_t, x_0)
                # Inpaint: replace the first datapoint with the condition
                # x_t[:, : , :self.inpaint_horizon, :] = x_0[:, : , :self.inpaint_horizon, :] # inpaint 
                # x_t[:, :, :, 2] = torch.clip(x_t[:, :, :, 2], min=-1.0, max=1.0) # Enforce action limits (steering angle)
                # x_t[:, :, :, 3:] = torch.clip(x_t[:, :, :, 3:], min=0.0, max=1.0)   # Enforce action limits (acceleration and brake)
                
                sampling_history.append(x_t.squeeze().detach().cpu().numpy())
            self.plt_toVideo(
                sampling_history,
                positions_groundtruth = positions_groundtruth,
                position_observation = position_observation)

         # q(x_t | x_0)
    def q_forwardProcess(self, x_start, t, noise):
        """
        x_start: (batch_size, pred_horizon, pred_dim)
        t: timestep (batch_size, 1)

        returns: x_t (batch_size, pred_horizon, pred_dim)
        """
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
            # # Inpaint: replace the first datapoint with the condition
            # x_t[:, : , :self.inpaint_horizon, :] = x_0[:, : , :self.inpaint_horizon, :] # inpaint 
            # x_t[:, :, :, 2] = torch.clip(x_t[:, :, :, 2], min=-1.0, max=1.0) # Enforce action limits (steering angle)
            # x_t[:, :, :, 3:] = torch.clip(x_t[:, :, :, 3:], min=0.0, max=1.0)   # Enforce action limits (acceleration and brake)

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
        # Adding constraints by inpainting before denoising. 
        # Add all constaints here
        x_t[:, : , :self.inpaint_horizon, :] = x_0[:, : , :self.inpaint_horizon, :].clone() # inpaint the first datapoint (should be enough)
        x_t[:, :, :, 2] = torch.clip(x_t[:, :, :, 2].clone(), min=-1.0, max=1.0) # Enforce action limits (steering angle)
        x_t[:, :, :, 3:] = torch.clip(x_t[:, :, :, 3:].clone(), min=0.0, max=1.0)   # Enforce action limits (acceleration and brake)
        return x_t

    # ==================== Helper functions ====================
    def prepare_pred_cond_vectors(self, batch):
        # Security check for corrupted data
        assert(not torch.isnan(batch['position'][:,: self.obs_horizon ,:]).any())
        assert(not torch.isnan(batch['action'][:,self.obs_horizon : ,:]).any())

        # ---------------- Preparing Observation data ----------------
        normalized_img = batch['image'][:,:self.obs_horizon ,:] 
        normalized_pos = batch['position'][:,:self.obs_horizon ,:]
        normalized_act = batch['action'][:,:self.obs_horizon ,:]
        # normalized_vel = batch['velocity'][:,:self.obs_horizon ,:]

        # ---------------- Encoding Image data ----------------
        encoded_img = self.vision_encoder(normalized_img.flatten(end_dim=1)) # (B, 512)
        image_features = encoded_img.reshape(*normalized_img.shape[:2],-1) # (B, t_0:t_obs , 512)

        # ---------------- Conditional vector ----------------
        # Concatenate position and action data and image features
        obs_cond = torch.cat([normalized_pos, normalized_act , image_features], dim=-1) # (B, t_0:t_obs, 512 + 3 + 2)

        # ---------------- Preparing Prediction data (acts as ground truth) ----------------
        x_0_pos = batch['position'][:,self.obs_horizon: ,:] # (B, t_obs:t_pred , 2)
        x_0_act = batch['action'][:, self.obs_horizon: ,:] # (B, t_obs:t_pred, 3)
        x_0 = torch.cat([x_0_pos, x_0_act], dim=-1) # (B, t_obs:t_pred, 5)

        # Adding past obervation as inpainting condition
        x_0 = torch.cat((obs_cond[:, -self.inpaint_horizon:, :5], x_0) , dim=1) # Concat in time dim
        return x_0 , obs_cond

    
    def plt_toTensorboard(self,
        position_observation,   
        positions_groundtruth,
        positions_predicted,
        actions_predicted,
        actions_groundtruth,
        actions_observation):
        # ---------------- Plotting ----------------
        writer = self.logger.experiment
        niter  = self.global_step

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

        plt.plot(position_observation[:, 0], position_observation[:,1],'b.')
        plt.plot(positions_groundtruth[self.inpaint_horizon:,0], positions_groundtruth[self.inpaint_horizon:,1],'g.')
        plt.scatter(positions_predicted[:,0],positions_predicted[:,1],color=colors, s = 20)

        plt.grid()
        plt.axis('equal')

        # Plot to tensorboard
        plt2tsb(fig, writer, 'Predicted_path ' + self.date , niter)

        # ---------------- Action space Plotting ----------------
        # Visualize the action data
        inpaint_start = 0
        inpaint_end = self.inpaint_horizon

        fig2, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.plot(actions_predicted[:,0])
        ax1.plot(actions_groundtruth[:,0])
        ax1.axvspan(inpaint_start, inpaint_end, alpha=0.3, color='red')

        ax2.plot(actions_predicted[:,1])
        ax2.plot(actions_groundtruth[:,1])
        ax2.axvspan(inpaint_start, inpaint_end, alpha=0.3, color='red')

        ax3.plot(actions_predicted[:,2])
        ax3.plot(actions_groundtruth[:,2])
        ax3.axvspan(inpaint_start, inpaint_end, alpha=0.3, color='red')

        plt2tsb(fig2, writer, 'Action comparisons' + self.date , niter)

        plt.close('all')


    def plt_toVideo(self, 
                    sampling_history,
                    position_observation,   
                    positions_groundtruth):

        sampling_positions = np.array(sampling_history)[:, :, :2]
        sampling_actions = np.array(sampling_history)[:, :, 2:]

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Create a colormap for fading colors based on the number of timesteps
        cmap = plt.get_cmap('viridis', self.pred_horizon + self.inpaint_horizon)

        # Create an array of indices from 0 to timesteps-1
        indices = np.arange(self.pred_horizon + self.inpaint_horizon)

        def animate(frame):
            fig.clf()
            # Normalize the indices to the range [0, 1]
            normalized_indices = indices / (self.pred_horizon + self.inpaint_horizon - 1)
            # Create a color array using the colormap and normalized indices
            colors = cmap(normalized_indices)

            plt.plot(position_observation[:, 0], position_observation[:, 1], 'b.')
            plt.plot(positions_groundtruth[self.inpaint_horizon:, 0], positions_groundtruth[self.inpaint_horizon:, 1], 'g.')
            plt.scatter(sampling_positions[frame, :, 0], sampling_positions[frame, :, 1], color=colors, s=20)

            plt.grid()
            plt.axis('equal')
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)


        # Create an animation using the FuncAnimation class
        animation = FuncAnimation(fig, animate, frames=self.noise_steps, interval=20, repeat=False)

        # Save the animation as a GIF
        animation.save('./animations/animation.gif', writer='pillow')

        # # Show the final plot
        plt.show()


# ==================== Utils ====================
def plt2tsb(figure, writer, fig_name, niter):
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)

    # Open the image and convert to RGB, then to Tensor
    image = Image.open(buf).convert('RGB')
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)

    # Add the image to TensorBoard
    writer.add_image(fig_name, image_tensor, niter)
    buf.close()

def linear_beta_schedule(self, steps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    beta = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32, device=self.device)
    return beta
    #return torch.cat([torch.tensor([0],device=self.device), beta])   # at t=0, we want beta=0.0, zero noise added


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

def print_hyperparameters(obs_horizon, pred_horizon, observation_dim, prediction_dim, noise_steps, inpaint_horizon, model, learning_rate, vision_encoder):
    print("*" * 20 + " OVERVIEW " +  "*" * 20)
    print()
    print("======== Hyperparameters =======")
    print("Date:", datetime.today().strftime('%Y-%m-%d-%H'))
    print("Observation Horizon:", obs_horizon)
    print("Prediction Horizon:", pred_horizon)
    print("Observation Dimension:", observation_dim)
    print("Prediction Dimension:", prediction_dim)
    print("Noise Steps:", noise_steps)
    print("Inpaint Horizon:", inpaint_horizon)

    print("======== Model Architecture =======")
    if model == 'UNet_Film':
        print("Model: UNet with FiLm conditioning")
    else:
        print("Model: UNet (simple)")
    print(" Learning Rate:", learning_rate)

    print("======== Model =======")
    print("Noise Estimator Model:")
    print("- In Channels:", 1)
    print("- Out Channels:", 1)
    print("- Noise Steps:", noise_steps)
    print("- Global Conditioning Dimension:", (observation_dim) * obs_horizon)
    print("- Time Dimension:", 256)

    print("Vision Encoder Model:")
    if vision_encoder == 'resnet18':
        print("Vision Encoder: Resnet18")
    else:
        print("Vision Encoder: Lightweight Autoencoder")
    print()
    print("*" * 40)