""" 

This file is used to run the environment and test the trained policy of my diffusion model.

I want the following features: 
    - The environment should be able to run with a set seed and controller / behavior policy
    - The data of the run is safed and can be called / visualized later
    - Loading of trained diffusion model and running it in the environment
    - Switching of the controller / behavior policy during run at set steps (e.g. 350/1000 steps)
"""


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
        data = pickle.load(file)
    return data



def load_model(path_hyperparams, path_checkpoint):
    model = Diffusion_DDPM.load_from_checkpoint(
        path_checkpoint,
        hparams_file=path_hyperparams,
        denosing_steps=250
    )
    model.eval()

    model_params = fetch_hyperparams_from_yaml(path_hyperparams)
    obs_horizon = model_params['obs_horizon']
    pred_horizon = model_params['pred_horizon']

    return model, obs_horizon, pred_horizon





# ======================  INITIALIZATION  ====================== #
seed = 1000
env = CarRacing()

  #parameters
start_steps = 420 

# -----  Load trained model  ----- #
path_hyperparams = './tb_logs/version_590/hparams.yaml'
path_checkpoint = './tb_logs/version_590/checkpoints/epoch=17.ckpt'
model, obs_horizon, pred_horizon = load_model(path_hyperparams, path_checkpoint)

file_path = 'MinMax.pkl'
stats = load_pickle_file(file_path)

# ----- Error buffers ----- #
error_hist = []
error_avg_hist = []
error_velocity_buffer = deque(np.zeros(7), maxlen = 7)
error_buffer = deque(np.zeros(10), maxlen = 10) # Buffer for storing errors for PID controller
error_buffer_2 = deque(np.zeros(3), maxlen = 3) # Buffer for storing errors for PID controller

#----- PID controllers ----- #
pid_velocity = PID(0.005, 0.001, 0.0005, setpoint=20) # PID(0.01, 0, 0.05, setpoint=0.0, output_limits=(0, 1))
pid_steering = PID(0.8, 0.01, 0.06, setpoint=0) # If negative switch over to breaking pedal

# -----  Run parameters ----- #
strip_distance = 60 # x - coordinates of the strip
car_pos_vector = np.array([70, 48])

# ----- Buffer for storing data ----- #
pos_buffer_obs = deque( obs_horizon * np.zeros(2), maxlen = obs_horizon) # Buffer for storing car position
image_buffer_obs = deque( obs_horizon * np.zeros((96, 96, 3)), maxlen = obs_horizon) # Buffer for storing image
velocity_buffer_obs = deque( obs_horizon * np.zeros(2), maxlen = obs_horizon) # Buffer for storing velocity
action_buffer_obs = deque( obs_horizon * np.zeros(3), maxlen = obs_horizon) # Buffer for storing action

# ----- Buffer for storing data ----- #
pos_buffer_pred = deque( pred_horizon * np.zeros(2), maxlen = pred_horizon) # Buffer for storing car position
image_buffer_pred = deque( pred_horizon * np.zeros((96, 96, 3)), maxlen = pred_horizon) # Buffer for storing image
velocity_buffer_pred = deque( pred_horizon * np.zeros(2), maxlen = pred_horizon) # Buffer for storing velocity
action_buffer_pred = deque( pred_horizon * np.zeros(3), maxlen = pred_horizon) # Buffer for storing action



# ======================  RUN ENVIRONMENT  ====================== #
# Run twice
for run in range(2):


  action = np.array([0, 0, 0], dtype=np.float32)
  env.seed(seed)
  env.reset()
  obs, _ , done, info = env.step(action) # Take a step to get the environment started (action is empty)
  start = time.time()
  steps = 0
  while not done:
    isopen = env.render("human")
    augmImg = info['augmented_img'] # Augmented image with colored trajectories
    velB2vec = info['car_velocity_vector']
    posB2vec = info['car_position_vector']  

    carVelocity_wFrame = [velB2vec.x , velB2vec.y]
    carPosition_wFrame = [posB2vec.x , posB2vec.y]
    v_wFrame = np.linalg.norm(velB2vec)

    if time.time() - start < 1: 
      # We wait for 1 second
      action = np.array([0.0, 0.0, 0.0])
      s, r, done, info = env.step(action) # Step has to be called for the environment to continue
      continue
    
      
    if steps < start_steps:
      # Use PD controller to move away from starting position
      action =trajectory_control(augmImg, 
                    strip_distance, 
                    car_pos_vector, 
                    pid_steering, 
                    pid_velocity, 
                    error_buffer,
                    error_buffer_2,
                    error_velocity_buffer,
                    v_wFrame,
                    "middle")
      
      obs, _, done, info = env.step(action)

      if run == 0:
        pos_buffer_obs.append(carPosition_wFrame)
        image_buffer_obs.append(obs)
        velocity_buffer_obs.append(carVelocity_wFrame)
        action_buffer_obs.append(action)
      
      steps += 1
      continue
    
    if run == 0:
      pos_buffer_obs = np.array(pos_buffer_obs, dtype=np.float32)
      image_buffer_obs = np.array(image_buffer_obs, dtype=np.float32)
      vel_buffer_obs = np.array(velocity_buffer_obs, dtype=np.float32)
      action_buffer_obs = np.array(action_buffer_obs, dtype=np.float32)
    
    
      # Continue to drive and saving predicted trajectory
      if steps < start_steps + pred_horizon:
        action_ =trajectory_control(augmImg, 
                      strip_distance, 
                      car_pos_vector, 
                      pid_steering, 
                      pid_velocity, 
                      error_buffer,
                      error_buffer_2,
                      error_velocity_buffer,
                      v_wFrame,
                      "middle")
        
        if action is not None:
          action = action_
           

        obs, _, done, info = env.step(action)
        
        pos_buffer_pred.append(carPosition_wFrame)
        image_buffer_pred.append(obs)
        velocity_buffer_pred.append(carVelocity_wFrame)
        action_buffer_pred.append(action)
        
        steps += 1
        continue
      
      pos_buffer_pred = np.array(pos_buffer_pred, dtype=np.float32)
      image_buffer_pred = np.array(image_buffer_pred, dtype=np.float32)
      vel_buffer_pred = np.array(velocity_buffer_pred, dtype=np.float32)
      action_buffer_pred = np.array(action_buffer_pred, dtype=np.float32)
      
      plt.plot(pos_buffer_obs[:,0], pos_buffer_obs[:,1], 'b')
      plt.plot(pos_buffer_pred[:,0], pos_buffer_pred[:,1], 'r')
      plt.show()
      
      break
    
    if run == 1:
      # Predict trajectory and actions: 
      
      nsample = {
        'image': image_buffer_obs,
        'position': torch.from_numpy(pos_buffer_obs),
        'velocity': torch.from_numpy(vel_buffer_obs),
        'action': torch.from_numpy(action_buffer_obs),
      }
      
      nsample_pred = {
        'image': torch.from_numpy(image_buffer_pred),
        'position': torch.from_numpy(pos_buffer_pred),
        'velocity': torch.from_numpy(vel_buffer_pred),
        'action': torch.from_numpy(action_buffer_pred),
      }
      
      pos_stats = stats[0]
      action_stat = stats[1]
      velocity_stat = stats[2]
      
      sample_normalized = normalize_data(nsample['position'], pos_stats)
      translation_vec = sample_normalized[0,:]
      nsample_centered = sample_normalized - translation_vec
      nsample['position'] = nsample_centered / 2.0
      
      sample_normalized = normalize_data(nsample_pred['position'], pos_stats)
      nsample_centered = sample_normalized - translation_vec
      nsample_pred['position'] = nsample_centered / 2.0
      
      plt.plot(nsample['position'][:,0], nsample['position'][:,1], 'b')
      plt.plot(nsample_pred['position'][:,0], nsample_pred['position'][:,1], 'r')
      plt.show()
      
      # Normalize the other data:
      nsample['velocity'] = normalize_data(nsample['velocity'], velocity_stat)
      nsample['action'] = normalize_data(nsample['action'], action_stat)
      
      # Adjust image data:
      # float32, [0,1], (N,96,96,3)
      nsample['image'] = nsample['image']/255.0
      nsample['image'] = np.moveaxis(nsample['image'], -1,1)
      nsample['image'] = torch.from_numpy(nsample['image'])

      model.sample(batch= nsample, mode='test')

    
    
    
  

