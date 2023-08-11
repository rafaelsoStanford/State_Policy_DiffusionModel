
import numpy as np
import torch

import time
from collections import deque
import matplotlib.pyplot as plt

from envs.car_racing import CarRacing
from utils.functions import *
from utils.load_data import *
from models.diffusion_ddpm import *

import pickle

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)[0]
    return data



# Model paths
filepath = './tb_logs/version_671/STATS.pkl'
path_checkpoint = './tb_logs/version_671/checkpoints/epoch=10.ckpt'
path_hyperparams = './tb_logs/version_671/hparams.yaml'

# Fetch parameters
model_params = fetch_hyperparams_from_yaml(path_hyperparams)
obs_horizon = model_params['obs_horizon']
pred_horizon = model_params['pred_horizon']
inpaint_horizon = model_params['inpaint_horizon']
step_size = model_params['step_size']

# Buffers for diffusion model
img_buffer_obs = deque(maxlen=obs_horizon)
pos_buffer_obs = deque(maxlen=obs_horizon)
vel_buffer_obs = deque(maxlen=obs_horizon)
act_buffer_obs = deque(maxlen=obs_horizon)

#Load ddpm model
ddpm_model = Diffusion_DDPM.load_from_checkpoint(
    path_checkpoint,
    hparams_file=path_hyperparams,
)
ddpm_model.eval()

# Load stats
stats = load_pickle_file(filepath)


# Environment parameters
env = CarRacing()
env.seed(42)
env.reset()
action = np.array([0, 0, 0], dtype=np.float32)
obs, _ , _, info = env.step(action) # Take a step to get the environment initialized (action is empty)

env.render("human")

while True:
    
    car_position = info['car_position_vector']
    px = car_position.x
    py = car_position.y

    position_points = [px, py]

    env.add_points2Buffer( position_points )
    action = np.array([0.1, 0.1, 0.0])
    # Get observation
    obs, _, done, info = env.step(action) # Step has to be called for the environment to continue
































# # ======================  RUN ENVIRONMENT  ====================== #
# # Run twice
# for run in range(2):


#   action = np.array([0, 0, 0], dtype=np.float32)
#   env.seed(seed)
#   env.reset()
#   obs, _ , done, info = env.step(action) # Take a step to get the environment started (action is empty)
#   start = time.time()
#   steps = 0
#   while not done:
#     isopen = env.render("human")
#     augmImg = info['augmented_img'] # Augmented image with colored trajectories
#     velB2vec = info['car_velocity_vector']
#     posB2vec = info['car_position_vector']  

#     carVelocity_wFrame = [velB2vec.x , velB2vec.y]
#     carPosition_wFrame = [posB2vec.x , posB2vec.y]
#     v_wFrame = np.linalg.norm(velB2vec)

#     if time.time() - start < 1: 
#       # We wait for 1 second
#       action = np.array([0.0, 0.0, 0.0])
#       s, r, done, info = env.step(action) # Step has to be called for the environment to continue
#       continue
    
      
#     if steps < start_steps:
#       # Use PD controller to move away from starting position
#       action =trajectory_control(augmImg, 
#                     strip_distance, 
#                     car_pos_vector, 
#                     pid_steering, 
#                     pid_velocity, 
#                     error_buffer,
#                     error_buffer_2,
#                     error_velocity_buffer,
#                     v_wFrame,
#                     "middle")
      
#       obs, _, done, info = env.step(action)

#       if run == 0:
#         pos_buffer_obs.append(carPosition_wFrame)
#         image_buffer_obs.append(obs)
#         velocity_buffer_obs.append(carVelocity_wFrame)
#         action_buffer_obs.append(action)
      
#       steps += 1
#       continue
    
#     if run == 0:
#       pos_buffer_obs = np.array(pos_buffer_obs, dtype=np.float32)
#       image_buffer_obs = np.array(image_buffer_obs, dtype=np.float32)
#       vel_buffer_obs = np.array(velocity_buffer_obs, dtype=np.float32)
#       action_buffer_obs = np.array(action_buffer_obs, dtype=np.float32)
    
    
#       # Continue to drive and saving predicted trajectory
#       if steps < start_steps + pred_horizon:
#         action =trajectory_control(augmImg, 
#                       strip_distance, 
#                       car_pos_vector, 
#                       pid_steering, 
#                       pid_velocity, 
#                       error_buffer,
#                       error_buffer_2,
#                       error_velocity_buffer,
#                       v_wFrame,
#                       "middle")
        
#         obs, _, done, info = env.step(action)
        
#         pos_buffer_pred.append(carPosition_wFrame)
#         image_buffer_pred.append(obs)
#         velocity_buffer_pred.append(carVelocity_wFrame)
#         action_buffer_pred.append(action)
        
#         steps += 1
#         continue
      
#       pos_buffer_pred = np.array(pos_buffer_pred, dtype=np.float32)
#       image_buffer_pred = np.array(image_buffer_pred, dtype=np.float32)
#       vel_buffer_pred = np.array(velocity_buffer_pred, dtype=np.float32)
#       action_buffer_pred = np.array(action_buffer_pred, dtype=np.float32)
      
#       plt.plot(pos_buffer_obs[:,0], pos_buffer_obs[:,1], 'b')
#       plt.plot(pos_buffer_pred[:,0], pos_buffer_pred[:,1], 'r')
#       plt.show()
      
#       break
    
#     if run == 1:
#       # Predict trajectory and actions: 
      
#       nsample = {
#         'image': image_buffer_obs,
#         'position': torch.from_numpy(pos_buffer_obs),
#         'velocity': torch.from_numpy(vel_buffer_obs),
#         'action': torch.from_numpy(action_buffer_obs),
#       }
      
#       nsample_pred = {
#         'image': torch.from_numpy(image_buffer_pred),
#         'position': torch.from_numpy(pos_buffer_pred),
#         'velocity': torch.from_numpy(vel_buffer_pred),
#         'action': torch.from_numpy(action_buffer_pred),
#       }
      
#       pos_stats = stats[0]
#       action_stat = stats[1]
#       velocity_stat = stats[2]
      
#       sample_normalized = normalize_data(nsample['position'], pos_stats)
#       translation_vec = sample_normalized[0,:]
#       nsample_centered = sample_normalized - translation_vec
#       nsample['position'] = nsample_centered / 2.0
      
#       sample_normalized = normalize_data(nsample_pred['position'], pos_stats)
#       nsample_centered = sample_normalized - translation_vec
#       nsample_pred['position'] = nsample_centered / 2.0
      
#       plt.plot(nsample['position'][:,0], nsample['position'][:,1], 'b')
#       plt.plot(nsample_pred['position'][:,0], nsample_pred['position'][:,1], 'r')
#       plt.show()
      
#       # Normalize the other data:
#       nsample['velocity'] = normalize_data(nsample['velocity'], velocity_stat)
#       nsample['action'] = normalize_data(nsample['action'], action_stat)
      
#       # Adjust image data:
#       # float32, [0,1], (N,96,96,3)
#       nsample['image'] = nsample['image']/255.0
#       nsample['image'] = np.moveaxis(nsample['image'], -1,1)
#       nsample['image'] = torch.from_numpy(nsample['image'])

#       model.sample(batch= nsample, mode='test')

    
    
    
  

