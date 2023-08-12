
import numpy as np
import torch

import time
from collections import deque
import matplotlib.pyplot as plt

from envs.car_racing import CarRacing
from utils.functions import *
from utils.load_data import *
from generateData.trajectory_control_utils import *
from models.diffusion_ddpm import *

import pickle

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)[0]
    return data



# ==================================================================================================
# Functions for running the environment
# ==================================================================================================



def prepare_diffusion_batch(img_buffer_obs, pos_buffer_obs, vel_buffer_obs, act_buffer_obs, stats):
    
    batch = {
        'image': torch.tensor(list(img_buffer_obs), dtype=torch.float32),
        'position': torch.tensor(list(pos_buffer_obs), dtype=torch.float32),
        'velocity': torch.tensor(list(vel_buffer_obs), dtype=torch.float32),
        'action': torch.tensor(list(act_buffer_obs), dtype=torch.float32),
    }

    normalized_image = batch['image'] / 255.0
    normalized_image = normalized_image.permute(0, 3, 1, 2)  # Change image to pytorch format
    normalized_action = normalize_data(batch['action'], stats['action'])
    normalized_velocity = normalize_data(batch['velocity'], stats['velocity'])
    normalized_position, position_translation = normalize_position(batch['position'], stats['position'])

    normalized_batch = { 
        'image': normalized_image.unsqueeze(0),
        'position': normalized_position.unsqueeze(0),
        'velocity': normalized_velocity.unsqueeze(0),
        'action': normalized_action.unsqueeze(0),
        'translation': position_translation.unsqueeze(0),
    }
    
    return normalized_batch , batch , position_translation


def get_initial_observation(obs, action,  info):
    augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)

    img = obs
    position_vector = np.array([carPosition_wFrame[0], carPosition_wFrame[1]])
    velocity_vector = np.array([carVelocity_wFrame[0], carVelocity_wFrame[1]])

    return img, position_vector, velocity_vector, action

# ==================================================================================================
# ==================================================================================================
# ==================================================================================================

# Model paths
filepath = './tb_logs/version_752/STATS.pkl'
path_checkpoint = './tb_logs/version_752/checkpoints/epoch=10.ckpt'
path_hyperparams = './tb_logs/version_752/hparams.yaml'

# Fetch parameters
model_params    = fetch_hyperparams_from_yaml(path_hyperparams)
obs_horizon     = model_params['obs_horizon']
pred_horizon    = model_params['pred_horizon']
inpaint_horizon = model_params['inpaint_horizon']
step_size       = model_params['step_size']

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

# Buffers for diffusion model
img_buffer_obs = deque(maxlen=obs_horizon)
pos_buffer_obs = deque(maxlen=obs_horizon)
vel_buffer_obs = deque(maxlen=obs_horizon)
act_buffer_obs = deque(maxlen=obs_horizon)

img, position_vector, velocity_vector, action = get_initial_observation(obs, action,  info)

for i in range(obs_horizon): # Fill buffers with initial observation
  img_buffer_obs.append(img)
  pos_buffer_obs.append(position_vector)
  vel_buffer_obs.append(velocity_vector)
  act_buffer_obs.append(action)

normalized_batch, batch, Translation = prepare_diffusion_batch(img_buffer_obs, pos_buffer_obs, vel_buffer_obs, act_buffer_obs, stats)

# Initialize controller 
velocity_error_buffer = deque(np.zeros(7), maxlen=7)
steering_error_buffer = deque(np.zeros(10), maxlen=10)
secondary_steering_error_buffer = deque(np.zeros(3), maxlen=3)

#----- PID controllers ----- #
velocity_pid_controller = PID(0.005, 0.001, 0.0005, setpoint=20)
steering_pid_controller = PID(0.8, 0.01, 0.06, setpoint=0)

env.render("human")

counter = 0
while True:    
    
    # Get environment variables
    augmented_image, car_velocity, car_position, car_heading_angle, velocity_in_frame = extract_variables(info)
    # Create Batch
    img_buffer_obs.append(img)
    pos_buffer_obs.append(car_position)
    vel_buffer_obs.append(car_velocity)
    act_buffer_obs.append(action)
    normalized_batch, batch , Translation = prepare_diffusion_batch(img_buffer_obs, pos_buffer_obs, vel_buffer_obs, act_buffer_obs, stats)
    # Prediction from diffusion model
    if counter%50 == 0:
      print("Predicting...")
      time_start = time.time()
      pred = ddpm_model.sample(normalized_batch, mode = "test")
      time_end = time.time()
      print("Done predicting. Result: {}    Elapsed Time: {}".format(pred.shape, time_end-time_start))

      # Extract prediction
      pred = pred[0].squeeze(0)
      pred = pred[inpaint_horizon:]
      pred = pred.cpu().detach().numpy()
      Translation = Translation.cpu().detach().numpy()
      # Unnormalize
      pred = unnormalize_position(pred, Translation , stats['position'])
      # Add point to buffer 
      points = pred
      env.add_points2Buffer(points)
      # print("Extracted prediction. Result: {}".format(pred.shape))


    # Calculate action
    calculated_action = trajectory_control(augmented_image, steering_pid_controller, velocity_pid_controller, 
                steering_error_buffer, secondary_steering_error_buffer, velocity_error_buffer, velocity_in_frame, 'left')
    
    # Take a step in the environment
    obs, reward, done, info = env.step(calculated_action)
    # p1 = np.array([car_position[0], car_position[1]])
    # p2 = np.array([car_position[0] + 0, car_position[1] + 10])
    # position_vector = np.array([p1,p2])
    # # Draw trajectory:
    # env.add_points2Buffer(position_vector)

    env.render("human")
    counter += 1
































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

    
    
    
  

