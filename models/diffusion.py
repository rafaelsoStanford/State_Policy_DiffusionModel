import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


import torch
import torch.nn as nn
from PIL import Image
import io


from models.Unet_FiLmLayer import *
from models.simple_Unet import * 

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
import pytorch_lightning as pl

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

def linear_beta_schedule(self, steps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    beta = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32, device=self.device)
    return torch.cat([torch.tensor([0],device=self.device), beta])   # at t=0, we want beta=0.0, zero noise added


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
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


class Diffusion(pl.LightningModule):
    def __init__(self, noise_steps=1000
                , obs_horizon = 10
                , pred_horizon= 10
                , observation_dim = 2
                , prediction_dim = 2
                , learning_rate = 1e-4
                , model = 'UNet'):
        
        super().__init__()

        # self.horizon = obs_horizon
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.observation_dim = observation_dim
        self.prediction_dim = prediction_dim
        self.noise_steps = noise_steps

        if model == 'UNet_Film':
            self.model = UNet_Film
        else:
            self.model = UNet
        
        betas = linear_beta_schedule(self, noise_steps)
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
        self.vision_encoder = VisionEncoder() # Loads pretrained weights of Resnet18 with output dim 512 (also modified layers as Suggested by Song et al.)
        self.vision_encoder.device = self.device
    
    def q_forwardProcess(self, x_start, t, noise=None): # q(x_t | x_{t-1})
        """
        x_start: (batch_size, pred_horizon, pred_dim)
        t: timestep (batch_size, 1)

        returns: x_t (batch_size, pred_horizon, pred_dim)
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device) # Needed if t = 1 (first step)
        #x_t = self.sqrt_alphas_cumprod[t][:, None, None]  * x_start + self.sqrt_one_minus_alphas_cumprod[t][:, None, None] * noise
        x_t = torch.sqrt(self.alphas_cumprod[t-1])[:, None, None]* x_start + torch.sqrt(1 - self.alphas_cumprod[t-1])[:, None, None] *noise
        return x_t

    
    # --------------------- Training / Validation ---------------------
    
    def onepass(self, batch, batch_idx, mode="train"):
        # batch = (B, tau_0 : tau_t, dim_obs)

        # Check if there are NaNs
        assert(not torch.isnan(batch['position'][:,: self.obs_horizon ,:]).any())

        # Observation data normalized to [-1,1]
        normalized_img = batch['image'][:,:self.obs_horizon ,:] # (B, tau_0 : tau_t, 3, 80, 80)
        normalized_pos = batch['position'][:,: self.obs_horizon ,:]
        # normalized_act = batch['action'][:,: self.obs_horizon ,:]

        # Encode image using Resnet18
        
        encoded_img = self.vision_encoder(normalized_img.flatten(end_dim=1)) # (B, 512)
        image_features = encoded_img.reshape(*normalized_img.shape[:2],-1) # (B, tau_0 : tau_t, 512)
        
        # Concatenate all features
        #obs_cond = normalized_pos.unsqueeze(1) # (B, 1, tau_0 : tau_t, pos_dim)
        
        
        obs_cond = torch.cat([normalized_pos, image_features ], dim=-1).unsqueeze(1) # (B, tau_0 : tau_t, 512 + pos_dim)
        #obs_cond = obs_features.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

        # Prediction data normalized to [-1,1] (Acts as ground truth)
        #x_0 = torch.cat( [batch['action'][:,self.obs_horizon: ,:], batch['position'][:,self.obs_horizon: ,:]], dim=-1)  #batch['position'][:,self.obs_horizon: ,:]
        x_0 = batch['position'][:,self.obs_horizon: ,:].unsqueeze(1) # (B, 1, pred_horizon, pred_dim)
        B = x_0.shape[0]

        # Add noise to prediction data
        noise = torch.randn_like(x_0)
        t = torch.randint(1, self.noise_steps+1, (B,), device=self.device).long() # Values from [1, 1000]
        # x_t = self.q_forwardProcess(x_0, t, noise=noise).unsqueeze(1) # (B, 1 , pred_horizon, pred_dim)
        x_noisy  = torch.sqrt(self.alphas_cumprod[t])[:,None,None,None] * x_0 + torch.sqrt(1-self.alphas_cumprod[t])[:,None,None,None] * noise

        # With conditioning using inpainting
        x_noisy[:, :, 0, :] = x_0[:, :, -1, :]

        # Estimate noise using noise_predictor
        if mode == "train":
            noise_estimated = self.noise_estimator(x_noisy, t, obs_cond )
        else:
            with torch.no_grad():
                noise_estimated = self.noise_estimator(x_noisy, t, obs_cond ) # (B, pred_horizon, pred_dim) 

        # Calculate loss
        loss = self.loss(noise_estimated, noise)
        return loss



    def training_step(self, batch, batch_idx):
        loss = self.onepass(batch, batch_idx, mode="train")
        self.log("train_loss",loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss
    
    
    def validation_step(self, batch, batch_idx):

        
        if batch_idx == 0:
            # # Observation data normalized to [-1,1]
            normalized_img = batch['image'][0,: self.obs_horizon , ...].unsqueeze(0)
            normalized_pos = batch['position'][0,: self.obs_horizon , :].unsqueeze(0)
            # normalized_act = batch['action'][0,: self.obs_horizon , :].unsqueeze(0)

            # # Encode image using Resnet18
            encoded_img = self.vision_encoder(normalized_img.flatten(end_dim=1)) # (B, 512)
            image_features = encoded_img.reshape(*normalized_img.shape[:2],-1) # (B, tau_0 : tau_t, 512)
            obs_cond = torch.cat([normalized_pos, image_features ], dim=-1).unsqueeze(1) # (B, tau_0 : tau_t, 512 + pos_dim)
            
            # Concatenate all features
            # obs_cond = normalized_pos[0,...].unsqueeze(0).unsqueeze(1) # (B, 1, tau_0 : tau_t, pos_dim)
            x_0 = batch['position'][0,self.obs_horizon: ,:].unsqueeze(0) # (B, pred_horizon, pred_dim)
            
            # Prediction data normalized to [-1,1] (Acts as ground truth)
            # x_0 = torch.cat(    [batch['action'][0,self.obs_horizon: ,:], 
            #                      batch['position'][0,self.obs_horizon: ,:]], dim=-1).unsqueeze(0)  # (B, 1, pred_horizon, pred_dim)
            #B = 1
            
            # Idea: Using the observations we predict the future action / position steps
            x_0_predicted = self.p_reverseProcess_loop(x_cond = obs_cond) # Only one batch is used to be visualized

            # Predictions
            positions_predicted = x_0_predicted.squeeze().detach().cpu().numpy() #(20 , 2)
            actions_predicted = x_0_predicted.squeeze().detach().cpu().numpy() #(20 , 3)

            # Observations ie Past
            position_observation = normalized_pos[0, ...].detach().cpu().numpy() 
            #actions_observation = normalized_act[0, ...].detach().cpu().numpy()

            positions_groundtruth = x_0[0, :].detach().cpu().numpy() #(20 , 2)
            actions_groundtruth = x_0[0,:].squeeze().detach().cpu().numpy() #(20 , 3)


            # Plot data
            writer = self.logger.experiment
            niter  = self.global_step

            # 2D position plot
            fig = plt.figure()
            fig.clf()

            # Create a colormap for fading colors based on the number of timesteps
            cmap = get_cmap('viridis', self.pred_horizon)
            # Create an array of indices from 0 to timesteps-1
            indices = np.arange(self.pred_horizon)
            # Normalize the indices to the range [0, 1]
            normalized_indices = indices / (self.pred_horizon - 1)
            # Create a color array using the colormap and normalized indices
            colors = cmap(normalized_indices)


            vals = np.linspace(1, 0.1, 10)

            plt.plot(position_observation[:, 0], position_observation[:,1],'b.')
            plt.plot(positions_groundtruth[:,0], positions_groundtruth[:,1],'g.')
            plt.scatter(positions_predicted[:,0],positions_predicted[:,1],color=colors, s = 10)

            plt.grid()
            plt.axis('equal')

            # Plot to tensorboard
            plt2tsb(fig, writer, 'Predicted_path', niter)


            # Visualize the action data
            # fig2, (ax1, ax2, ax3) = plt.subplots(1, 3)
            # ax1.plot(action_prediction[:,0])
            # ax1.plot(action_predicted[:,0])
            # # ax1.scatter( np.arange(train_data['action'].shape[0]), train_data['action'][:,0] , c='r', s=1)
            # ax2.plot(action_prediction[:,1])
            # ax2.plot(action_predicted[:,1])
            # # ax2.scatter( np.arange(train_data['action'].shape[0]), train_data['action'][:,1] , c='r', s=1)
            # ax3.plot(action_prediction[:,2])
            # ax3.plot(action_predicted[:,2])
            # # ax3.scatter( np.arange(train_data['action'].shape[0]), train_data['action'][:,2] , c='r', s=1)
            # plt2tsb(fig2, writer, 'Action comparisons', niter)

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

    # ---------------------- Sampling --------------------------------

    # Given noisy sample at time T, denoise to get sample at time 0
    # @torch.no_grad()
    # def p_reverseProcess_loop(self, x_cond, x_T = None):
    #     B = 1
    #     horizon = self.pred_horizon
    #     dim = self.prediction_dim  
    #     if x_T is None:
    #         x_T = torch.randn(B , 1 , horizon ,dim, device = self.device) # (B, 1 ,tau_0 : tau_t, 5)

    #     # Run loop for N = denoising_steps
    #     x_t = x_T
    #     for t in range(self.noise_steps , 0, -1): #  [1000, 999, ..., 1]
    #         noise_est = self.noise_estimator(x_t, torch.tensor(t, device= self.device),  x_cond)
    #         #x_t = self.p_reverseProcess(x_t, torch.tensor(t, device=self.device), x_cond) 
    #         if t == 1:
    #             x_t  = 1.0/torch.sqrt(self.alphas[t-1]) * ( x_t - (1.0-self.alphas[t-1])/torch.sqrt(1.0-self.alphas_cumprod[t-1])*noise_est )
    #         else:   
    #             z = torch.randn_like(x_t, device = self.device)
    #             x_t  = 1.0/torch.sqrt(self.alphas[t-1]) * (x_t-(1.0-self.alphas[t-1])/torch.sqrt(1.0-self.alphas_cumprod[t-1])*noise_est) + \
    #                     torch.sqrt(self.betas[t-1])*z
    #     return x_t

    @torch.no_grad()
    def p_reverseProcess_loop(self, x_cond, x_T = None):
        if x_T is None:
            x_t = torch.rand(1, 1, self.pred_horizon, self.prediction_dim, device=self.device)
        else:
            x_t = x_T
        
        for t in range(self.noise_steps ,0,-1):
            with torch.no_grad():                
                est_noise = self.noise_estimator(x_t, torch.tensor([t], device=self.device), x_cond)
            if t > 1:
                z = torch.randn_like(x_t)
                x_t = 1/torch.sqrt(self.alphas[t])* (x_t-(1-self.alphas[t])/torch.sqrt(1-self.alphas_cumprod[t])*est_noise) +  torch.sqrt(self.betas[t])*z

            else:
                x_t = 1/torch.sqrt(self.alphas[t])* (x_t-(1-self.alphas[t])/torch.sqrt(1-self.alphas_cumprod[t])*est_noise)

            # Inpaint: replace the first datapoint with the condition
            x_t[:, : , 0, :] = x_cond[:, : , -1, :2] # inpaint the first datapoint (should be enough)

        return x_t


# def test_forward_process():

#     from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
#     from PIL import Image

#     diffusion = Diffusion()
    
#     image_size = 128
#     transform = Compose([
#                 Resize(image_size),
#                 CenterCrop(image_size),
#                 ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
#                 Lambda(lambda t: (t * 2) - 1),

#     ])

#     reverse_transform = Compose([
#                         Lambda(lambda t: (t + 1) / 2),
#                         Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#                         Lambda(lambda t: t * 255.),
#                         Lambda(lambda t: t.numpy().astype(np.uint8)),
#                         ToPILImage(),
#                         ])

#     # Test forward pass
#     img = Image.open("./data/tests/test.jpg") # PIL image of shape HWC
#     t = torch.tensor(0)
#     x_start = transform(img).unsqueeze(0) #(1,3,128,128)
#     noise = torch.randn_like(x_start)
#     x_t = diffusion.q_forwardProcess(x_start, t ,noise)
#     noisy_img = reverse_transform(x_t.squeeze(0)) 
#     noisy_img.show()

# if __name__ == "__main__":
#     test_forward_process()