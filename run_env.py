""" 

This file is used to run the environment and test the trained policy of my diffusion model.

I want the following features: 
    - The environment should be able to run with a set seed and controller / behavior policy
    - The data of the run is safed and can be called / visualized later
    - Loading of trained diffusion model and running it in the environment
    - Switching of the controller / behavior policy during run at set steps (e.g. 350/1000 steps)
"""

import os
import subprocess
import numpy as np
import torch
import gym
import time
from collections import deque

import math
import matplotlib.pyplot as plt

from envs.car_racing import CarRacing
from utils.functions import *
from utils.load_data import *
from models.diffusion import *

# ======================  HELPER FUNCTIONS  ====================== #




# ======================  INITIALIZATION  ====================== #
seed = 1000
env = CarRacing()
env.seed(seed) 
env.reset()
state, reward, done, info = env.step(np.array([0, 0, 0]))

  #parameters
max_steps = 100  # 3000
target_velocity = 20

  #buffers
pos_hist = []
vel_hist = []
action_hist = []

 # diffusion model buffers
diffusion_pos_hist = []
diffusion_action_hist = []

# ======================  RUN ENVIRONMENT  ====================== #
for i in range(max_steps):
    observation = {
      "image": state,
      "velocity": np.linalg.norm(info['car_velocity_vector'])
    }
    print("Velocity: ", observation['velocity'])
    action = calculateAction(observation, target_velocity)
    state, reward, done, info = env.step(action)


    pos_hist.append(info['car_position_vector'].copy())
    action_hist.append(action.copy())
    vel_hist.append(info['car_velocity_vector'].copy())

    env.render()
    if done:
        break

# ======================  PLOT RESULTS  ====================== #
pos_hist = np.array(pos_hist)
action_hist = np.array(action_hist)

plt.figure()
plt.plot(pos_hist[:,0], pos_hist[:,1])
plt.title('Position of car')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# ======================  RUN DIFFUSION MODEL  ====================== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for Diffusion Model")
# ---------- Load Model ---------- #
path_hyperparams = './tb_logs/version_503/hparams.yaml'
path_checkpoint = './tb_logs/version_503/checkpoints/epoch=30.ckpt'

model_params = fetch_hyperparams_from_yaml(path_hyperparams)
model = Diffusion.load_from_checkpoint(
    path_checkpoint,
    hparams_file=path_hyperparams
)
model.eval() # set model to evaluation mode

# Diffusion model parameters
obs_horizon = model_params['obs_horizon']
pred_horizon = model_params['pred_horizon']
prediction_dim = model_params['prediction_dim']
inpaint_horizon = model_params['inpaint_horizon']
noise_steps = model_params['noise_steps']

# Reset environment
env.reset()
env.render()
image, reward, done, info = env.step(np.array([0, 0, 0]))

observation = { 
    "image": image,
    "position": info['car_position_vector'],
    "velocity": info['car_velocity_vector'].copy(),
    "action": np.array([0, 0, 0]),
}

# Initialize observation deque -- observation buffer
observation_deque = deque([observation]*obs_horizon ,maxlen=obs_horizon)

betas =  model.NoiseScheduler(model, noise_steps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

model.register_buffer('betas', betas)
model.register_buffer('alphas', alphas)
model.register_buffer('alphas_cumprod', alphas_cumprod)
# calculations for diffusion q(x_t | x_{t-1}) and others
model.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
model.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

steps = 0
action_horizon = 1
diff_visited_pos = []
for i in range(max_steps):
    # ---------- Get observation ---------- #
    # stack the last obs_horizon number of observations
  images = np.stack(x['image'] for x in observation_deque)
  images = np.moveaxis(images, -1,1)
  position =  np.stack(pos_hist[0, :] for _ in observation_deque)  #pos_hist[:obs_horizon,:]
  velocity = np.stack([vel_hist[0].x , vel_hist[0].y]  for _ in observation_deque)  #  vel_hist[:obs_horizon]
  action = np.stack(action_hist[0,:] for _ in observation_deque) # action_hist[:obs_horizon,:]

  # ---------- Normalization ---------- #
  position_stats = get_data_stats(position)
  velocity_stats = get_data_stats(velocity)
  normalized_position = torch.tensor(normalize_data(position, position_stats), dtype=torch.float32, device=device).unsqueeze(0) # (1, t_0:t_obs , 2)
  normalized_velocity = torch.tensor(normalize_data(velocity, velocity_stats), dtype=torch.float32, device=device).unsqueeze(0) # (1, t_0:t_obs , 1)
  normalized_images = torch.tensor(images / 255.0, dtype=torch.float32, device=device).unsqueeze(0) # (1, t_0:t_obs , 96, 96, 3)
  normalized_actions = torch.tensor(action, dtype= torch.float32, device=device).unsqueeze(0) # no normalization needed)

    # ---------------- Encoding Image data ----------------
  encoded_img = model.vision_encoder(normalized_images.flatten(end_dim=1)) # (B, 128)
  image_features = encoded_img.reshape(*normalized_images.shape[:2],-1) # (B, t_0:t_obs , 128)



  normalized_position = torch.nan_to_num(normalized_position, nan=0.0)
  normalized_velocity = torch.nan_to_num(normalized_velocity, nan=0.0)
  normalized_actions = torch.nan_to_num(normalized_actions, nan=0.0)

  assert(not torch.isnan(normalized_position).any())
  assert(not torch.isnan(normalized_velocity).any())
  assert(not torch.isnan(normalized_actions).any())

  # ---------- create conditioning vector ---------- #
  obs_cond = torch.cat((normalized_position, normalized_actions, normalized_velocity, image_features), dim=-1) # (B, 1, t_0:t_obs , 2+3+128)
  obs_inpaint = obs_cond[:, -inpaint_horizon:, :5] # (B, 1, t_obs-t_pred:t_obs , 2+3) # only position and actions

  # ---------- Initialize noise vector with output shape ---------- #
  x_0_noisy = torch.randn(
      (1, 1, pred_horizon + inpaint_horizon, prediction_dim), device=device)

  # ---------- Run diffusion model ---------- #
  obs_cond = obs_cond.unsqueeze(1)
  obs_inpaint = obs_inpaint.unsqueeze(1)
  x_t  = x_0_noisy
  with torch.no_grad():
    for t in reversed(range(0,noise_steps)):
      if t == 0:
          z = torch.zeros_like(x_t)
      else:
          z = torch.randn_like(x_t)
      est_noise = model.noise_estimator(x_t, torch.tensor([t], device=device), obs_cond)
      x_t = 1/torch.sqrt(model.alphas[t])* (x_t-(1-model.alphas[t])/torch.sqrt(1-model.alphas_cumprod[t])*est_noise) +  torch.sqrt(model.betas[t])*z

      #Inpainting
      x_t[:, : , :inpaint_horizon, :] = obs_inpaint.clone() # inpaint the first datapoint (should be enough)
      # x_t[:, :, :, 2] = torch.clip(x_t[:, :, :, 2].clone(), min=-1.0, max=1.0) # Enforce action limits (steering angle)
      # x_t[:, :, :, 3:] = torch.clip(x_t[:, :, :, 3:].clone(), min=-1.0, max=1.0)   # Enforce action limits (acceleration and brake)

  position_pred = x_t[0,0, inpaint_horizon:, :2].detach().cpu().numpy()
  action_pred = x_t[0,0, inpaint_horizon:, 2:].detach().cpu().numpy()
  
  for step in range(action_horizon):
    action = action_pred[step, :]
    state, reward, done, info = env.step(action)

    observation = { 
      "image": image,
      "position": info['car_position_vector'],
      "velocity": info['car_velocity_vector'],
      "action": action.copy(),
    }

    observation_deque.append(observation)
    diff_visited_pos.append([info['car_position_vector'].x , info['car_position_vector'].y ] .copy())
    diffusion_pos_hist.append(position_pred.copy())
    diffusion_action_hist.append(action_pred.copy())
 
    env.render()
    i = i + 1

  env.render()
  if done:
      break
env.close()

# ======================  PLOT RESULTS  ====================== #
# Plotting
plt.switch_backend('TkAgg')
unnormalized_position = unnormalize_data(np.array(diff_visited_pos), position_stats)

plt.plot(unnormalized_position[:, 0], unnormalized_position[:, 1], label='Diffusion')
plt.plot(pos_hist[:, 0], pos_hist[:, 1], label='Ground Truth')
plt.legend()
plt.show()
