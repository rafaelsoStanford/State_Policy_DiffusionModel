'''
Using conda environement gym_0.21.0
Copied  implmentation of ReplayBuffer from diffusion policy  into the Utils folder.
This ensures that the replay buffer is compatible with their environment when training diffusion in Colab file.

IMPORTANT:  The current environment of car_race is not compatible with gym or gymnasium (which would be version 0.26.0).
            It is compatible with gym==0.21.0!!! 
            This is because the car_racing.py file was modified and doing it again for the new version of gym would be a pain.
            Maybe I will do it in the future, but for now, I will stick with gym==0.21.0
'''

import os
import sys
import shutil
import zarr
import numpy as np
from datetime import datetime
import argparse

# setting path
sys.path.append('../diffusion_bare')

from utils.replay_buffer import ReplayBuffer
from envs.car_racing import CarRacing
from utils.functions import *

global render_mode # Global variable to render the environment

def pidDriver(env, buffer, TARGET_VELOCITY, NUM_EPISODES):
    for episode in range(NUM_EPISODES):
        print("Episode: ", episode)

        # Initialize history lists
        img_hist, vel_hist, act_hist, pos_hist = [], [], [], []

        # Reset the environment
        obs = env.reset()
        action = np.array([0.0, 0.0, 0.0])
        obs, _, done, info = env.step(action)

        max_iter = 1000
        iter = 0
        done = False

        while not done:
            env.render(mode=render_mode)

            # Get velocity and position vectors in the world frame
            velB2vec = info['car_velocity_vector']
            posB2vec = info['car_position_vector']
            carVelocity_wFrame = [velB2vec.x, velB2vec.y]
            carPosition_wFrame = [posB2vec.x, posB2vec.y]
            v_wFrame = np.linalg.norm(velB2vec)

            # Create observation dictionary
            observation = {
                "image": obs,
                "velocity": v_wFrame,
            }

            # Calculate action using PID controller
            action = calculateAction(observation, TARGET_VELOCITY)

            action[0] = np.clip(action[0], -1, 1)
            action[1] = np.clip(action[1], 0, 1)
            action[2] = np.clip(action[2], 0, 1)
            
            # Take the action
            obs, reward, done, info = env.step(action)

            # Save the observation and action
            img_hist.append(obs)
            vel_hist.append(carVelocity_wFrame)
            pos_hist.append(carPosition_wFrame)
            act_hist.append(action.copy())

            if iter == max_iter:
                done = True
            iter += 1

        # Convert history lists to numpy arrays
        img_hist = np.array(img_hist, dtype=np.float32)
        act_hist = np.array(act_hist, dtype=np.float32)
        vel_hist = np.array(vel_hist, dtype=np.float32)
        pos_hist = np.array(pos_hist, dtype=np.float32)

        # Normalize each image in img_hist to be between 0 and 1
        img_hist = img_hist / 255.0

        # Check if act_hist has NaN values. If yes, replace with 0
        if np.isnan(act_hist).any():
            act_hist = np.nan_to_num(act_hist)
            print(" WARNING: act_hist had NaN values. Replaced with 0")

        # Create episode data dictionary
        episode_data = {
            "img": img_hist,
            "velocity": vel_hist,
            "position": pos_hist,
            "action": act_hist,
            "h_action": act_hist  # This will act as a placeholder for "human action". It is crude, but works for current testing purposes
        }

        buffer.add_episode(episode_data)

        print("Episode finished after {} timesteps".format(len(img_hist)))
        env.close()

    return img_hist, vel_hist, act_hist, pos_hist

        
def sinusoidalDriverSafe(env, buffer, TARGET_VELOCITY, NUM_EPISODES):
    for episode in range(NUM_EPISODES):
        print("Episode: ", episode)
        
        obs = env.reset()
        action = np.array([0.0, 0.0, 0.0])
        obs, _ , done, info = env.step(action)

    #  -------------------  Params ------------------- #
        Amplitude = 5 #Safe (ie within bounds of track); found by trial and error
        freq = 1/100 
        done = False
        max_iter = 1000
        iter = 0
        img_hist, vel_hist ,act_hist, pos_hist = [], [], [], []

        while not done:
            env.render(mode=render_mode)

            velB2vec = info['car_velocity_vector']
            posB2vec = info['car_position_vector']  

            carVelocity_wFrame = [velB2vec.x , velB2vec.y]
            carPosition_wFrame = [posB2vec.x , posB2vec.y]
            v_wFrame = np.linalg.norm(velB2vec)

            observation = {
                "image": obs,
                "velocity": v_wFrame, 
            }

            action = action_sinusoidalTrajectory(iter, freq, observation, Amplitude ,TARGET_VELOCITY)
            
            # Save the observation and action            
            img_hist.append(obs)
            vel_hist.append(carVelocity_wFrame)
            pos_hist.append(carPosition_wFrame)
            act_hist.append(action.copy())

            # Take the action
            obs, reward, done, info = env.step(action)
        
            if iter == max_iter:
                done = True
            iter += 1
        
        # ======================  SAVE DATA  ====================== #
        img_hist = np.array(img_hist, dtype=np.float32)
        act_hist = np.array(act_hist, dtype=np.float32)
        vel_hist = np.array(vel_hist, dtype=np.float32)
        pos_hist = np.array(pos_hist, dtype=np.float32)

        # Normalize each image in img_hist to be between 0 and 1
        img_hist = img_hist / 255.0

        # Check if act_hist has nan values. If yes, replace with 0
        if np.isnan(act_hist).any():
            act_hist = np.nan_to_num(act_hist)
            print(" WARNING: act_hist had nan values. Replaced with 0")

        episode_data = {
                "img": img_hist, 
                "velocity": vel_hist, 
                "position": pos_hist,
                "action": act_hist, 
                "h_action": act_hist #This will act as a placeholder for "human action". It is crude, but works for current testing purposes
                }

        buffer.add_episode(episode_data)

        print("Episode finished after {} timesteps".format(len(img_hist)))
        env.close()
    return img_hist, vel_hist ,act_hist, pos_hist

def sinusoidalDriverUnsafe(env, buffer, TARGET_VELOCITY, NUM_EPISODES):
    for episode in range(NUM_EPISODES):
        print("Episode: ", episode)
        
        obs = env.reset()
        action = np.array([0.0, 0.0, 0.0])
        obs, _ , done, info = env.step(action)

    #  -------------------  Params ------------------- #
        Amplitude = 13 #Unsafe (ie outside bounds of track); found by trial and error
        freq = 1/100 
        done = False
        max_iter = 1000
        iter = 0
        img_hist, vel_hist ,act_hist, pos_hist = [], [], [], []

        while not done:
            env.render(mode=render_mode)

            velB2vec = info['car_velocity_vector']
            posB2vec = info['car_position_vector']  

            carVelocity_wFrame = [velB2vec.x , velB2vec.y]
            carPosition_wFrame = [posB2vec.x , posB2vec.y]
            v_wFrame = np.linalg.norm(velB2vec)

            observation = {
                "image": obs,
                "velocity": v_wFrame, 
            }

            action = action_sinusoidalTrajectory(iter, freq, observation, Amplitude ,TARGET_VELOCITY)
            
            # Save the observation and action            
            img_hist.append(obs)
            vel_hist.append(carVelocity_wFrame)
            pos_hist.append(carPosition_wFrame)
            act_hist.append(action.copy())

            # Take the action
            obs, reward, done, info = env.step(action)
        
            if iter == max_iter:
                done = True
            iter += 1
        
        # ======================  SAVE DATA  ====================== #
        img_hist = np.array(img_hist, dtype=np.float32)
        act_hist = np.array(act_hist, dtype=np.float32)
        vel_hist = np.array(vel_hist, dtype=np.float32)
        pos_hist = np.array(pos_hist, dtype=np.float32)

        # Normalize each image in img_hist to be between 0 and 1
        img_hist = img_hist / 255.0

        # Check if act_hist has nan values. If yes, replace with 0
        if np.isnan(act_hist).any():
            act_hist = np.nan_to_num(act_hist)
            print(" WARNING: act_hist had nan values. Replaced with 0")

        episode_data = {
                "img": img_hist, 
                "velocity": vel_hist, 
                "position": pos_hist,
                "action": act_hist, 
                "h_action": act_hist #This will act as a placeholder for "human action". It is crude, but works for current testing purposes
                }

        buffer.add_episode(episode_data)

        print("Episode finished after {} timesteps".format(len(img_hist)))
        env.close()
    return img_hist, vel_hist ,act_hist, pos_hist

def generateData(args):
    # ======================  PARAMETERS  ====================== #
    NUM_EPISODES_PER_EPISODE = args.num_episodes
    CHUNK_LEN = args.chunk_len
    dataset_name = args.dataset_name
    base_dir = args.base_dir
    velocity = args.velocity 

    # ======================  INITIALIZATION  ====================== #
    # Init environment and buffer
    env = CarRacing()
    buffer = ReplayBuffer.create_empty_numpy()

    # ======================  GENERATE DATA  ====================== #
    pidDriver(env, buffer,velocity, NUM_EPISODES_PER_EPISODE)
    sinusoidalDriverSafe(env , buffer, velocity, NUM_EPISODES_PER_EPISODE)
    sinusoidalDriverUnsafe(env, buffer, velocity, NUM_EPISODES_PER_EPISODE)

    # ======================  SAVE DATA  ====================== #
    today = datetime.now()    # Get today's date
    folder_name = today.strftime("%Y-%m-%d-%H%M") # Format the date as a string
    # Check if the folder exists
    if not os.path.exists(base_dir + folder_name):
        # Create the folder
        os.makedirs(base_dir + folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists.")

    if dataset_name:
        dataset_name = dataset_name.rstrip('.zarr')
        dataset_name = dataset_name + f"_dataset_{NUM_EPISODES_PER_EPISODE}_episodes.zarr"
    else:
        dataset_name = folder_name + f"_dataset_{NUM_EPISODES_PER_EPISODE}_episodes.zarr"
    path = os.path.join(base_dir, folder_name, dataset_name)

    print("Saving data to path: ", path)
    buffer.save_to_path(path, chunk_length=CHUNK_LEN)

    # Consolidate metadata and zip the file
    store = zarr.DirectoryStore(path)
    data = zarr.group(store=store)
    print(data.tree(expand=True))
    zarr_file = os.path.basename(path)
    zip_file = zarr_file + ".zip"
    zarr.consolidate_metadata(store)
    shutil.make_archive(path, "zip", path)
    print(f"Zarr file saved as {zip_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Data")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes per mode")
    parser.add_argument("--chunk_len", type=int, default=-1, help="Chunk length")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name")
    parser.add_argument("--base_dir", type=str, default="./data/", help="Base directory")
    parser.add_argument("--velocity", type=int, default=30, help="Target velocitiy for the car")
    parser.add_argument("--render_mode", type=str, default="human", help="render mode of gym env. human means render, rgb_array means no render visible")

    args = parser.parse_args()
    render_mode = args.render_mode

    generateData(args)
