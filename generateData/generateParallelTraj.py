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
import argparse

# setting path
sys.path.append('../diffusion_bare')

from utils.replay_buffer import ReplayBuffer
from envs.car_racing import CarRacing
from utils.functions import *

import random
from tqdm.auto import trange
from datetime import datetime

global render_mode 

def findClosestPoint(trajectory_img, carPos = np.array([70, 48])):
    trajectory_idx = np.nonzero(trajectory_img)
    #Find single closest edge point
    distanceCarToTraj = np.linalg.norm(np.array(carPos)[:, None] - np.array(trajectory_idx), axis=0)
    closetPointIdx = np.argmin(distanceCarToTraj)
    closestPoint = np.array([trajectory_idx[0][closetPointIdx], trajectory_idx[1][closetPointIdx]])
    return closestPoint

def maskTrajecories(image):

    # Define the threshold ranges for each mask
    lower_yellow = np.array([100, 100, 0], dtype=np.uint8) ## Left of track in grass
    upper_yellow = np.array([255, 255, 0], dtype=np.uint8)
    lower_cyan = np.array([0, 100, 100], dtype=np.uint8) ## Left side of track on road
    upper_cyan = np.array([0, 255, 255], dtype=np.uint8)
    lower_magenta = np.array([100, 0, 100], dtype=np.uint8) ## Middle of track on road
    upper_magenta = np.array([255, 0, 255], dtype=np.uint8)
    lower_purple = np.array([100, 0 , 100], dtype=np.uint8) ## Right side of track on road
    upper_purple = np.array([200, 50,200], dtype=np.uint8)
    lower_blue = np.array([0, 0, 100], dtype=np.uint8) ## Right of track in grass
    upper_blue = np.array([0, 0, 255], dtype=np.uint8)

    # Apply the masks
    mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(image, lower_blue, upper_blue)
    mask_cyan = cv2.inRange(image, lower_cyan, upper_cyan)
    mask_magenta = cv2.inRange(image, lower_magenta, upper_magenta)
    mask_purple = cv2.inRange(image, lower_purple, upper_purple)

    # Label the differently colored trajectories
    dict_masks = {'lleft': mask_yellow, 
                  'left': mask_cyan, 
                  'middle': mask_magenta, 
                  'right': mask_purple, 
                  'rright': mask_blue}
    return dict_masks

def switch_mode(modes):
    chosen_mode = random.choice(modes)
    return chosen_mode

def driving(env, buffer, NUM_EPISODES, MODE, VELOCITIES):
    # ======================  INITIALIZATION  ====================== #
    # Random seed for each episode -- track during episode is different
    seeds = np.random.randint(0, 200, size=NUM_EPISODES)
    # PID controllers for velocity, steering and breaking
    pid_velocity = PID(0.01, 0, 0.05, setpoint=0.0, output_limits=(0, 1)) # PID(0.01, 0, 0.05, setpoint=0.0, output_limits=(0, 1))
    pid_breaking = PID(0.05, 0.00, 0.08, setpoint=0.0, output_limits=(0, 0.9))
    pid_steering = PID(0.8, 0.01, 0.3, setpoint=0, output_limits=(-1, 1))  # PID(0.5, 0.01, 100.0, setpoint=0, output_limits=(-1, 1))
    # Image car coordinates ( fixed in image frame)
    car_pos_vector = np.array([70, 48])
    max_steps = 1000 # Max steps per episode
    wait_steps = 100 # Wait steps before starting to collect data
    velocitites = VELOCITIES

    # ======================  START RUN  ====================== #
    print("*"*10 +" Starting run...: Current Mode: ", MODE, "*"*10)
    for episode in trange(NUM_EPISODES, desc="Episode"):
        #Initialize list for storing data -- will be sent to zarr replay buffer
        img_hist, vel_hist ,act_hist, pos_hist = [], [], [], []
        #Initialize action array
        action = np.array([0, 0, 0], dtype=np.float32)
        #Initialize environment
        env.seed(int(seeds[episode]))
        env.reset()
        obs, _ , done, info = env.step(action) # Take a step to get the environment started (action is empty)

        for i in trange(max_steps + wait_steps, desc="Step"):
            isopen = env.render(render_mode)
            augmImg = info['augmented_img'] # Augmented image with colored trajectories
            velB2vec = info['car_velocity_vector']
            posB2vec = info['car_position_vector']  

            carVelocity_wFrame = [velB2vec.x , velB2vec.y]
            carPosition_wFrame = [posB2vec.x , posB2vec.y]
            v_wFrame = np.linalg.norm(velB2vec)
            
            if i < wait_steps: # First 100 steps, we do nothing (avoid zooming animation)
                env.step(action)
                continue

            if i % 200 == 0: # All 200 steps, we change velocity
                pid_velocity.setpoint = random.choice(velocitites)

            # ======================  TRAJECTORY CONTROL  ====================== #
            # Render all trajectories using masks:
            dict_masks = maskTrajecories(augmImg)
            track_img = dict_masks[MODE] # Get the correct mask for the desired agent
            # Get single line strip in front of car
            line_strip = track_img[60, :]
            idx = np.nonzero(line_strip)[0]

            if len(idx) == 0: # Rarely happens, but sometimes the is no intersection of trajectory with line strip -> continue with previous action
                action[0] = -1.0
                obs, _, done, info = env.step(action)
                continue

            # Get index closest to middle of strip (idx = 48)
            idx = idx[np.argmin(np.abs(idx - 48))]
            target_point = np.array([60, idx])
            car2point_vector = target_point - car_pos_vector# As an approximation let angle be the x component of the car2point vector
            
            # ======================  PID CONTROL  ====================== #
            # Use distance and velocity information to control the car. Some values are hardcoded, by trial and error
        
            err =  idx - 48 # Correcting for the fact that negative value is a left turn, ie positive angle
            err = np.clip(err, -5, 5) # Clip the error to avoid large changes of steering angle going to infinity

            angle = np.arctan2(abs(err), abs(car2point_vector[0]))
            if err > 0:
                angle = -angle
            action[0] = pid_steering(angle)

            # if abs(action[0]) > 0.8:
            #     action[2] = 0.9

            if pid_velocity.setpoint - np.linalg.norm(v_wFrame) < -5:
                action[1] = 0
                action[2] = np.clip(pid_breaking(np.linalg.norm(v_wFrame)), 0, 0.9)
            else:
                action[1] = np.clip(pid_velocity(np.linalg.norm(v_wFrame)), 0, 1.0)
                action[2] = 0

            obs, _ , done, info = env.step(action)
            
            # Save the observation and action            
            img_hist.append(obs)
            vel_hist.append(carVelocity_wFrame)
            pos_hist.append(carPosition_wFrame)
            act_hist.append(action.copy())
            
            # Take the action
            obs, _, done, info = env.step(action)
            if done or isopen == False:
                break
        
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
    NUM_EPISODES_PER_MODE = args.num_episodes_per_mode
    CHUNK_LEN = args.chunk_len
    dataset_name = args.dataset_name
    base_dir = args.base_dir
    modes = args.modes #['middle', 'left', 'right'] #['lleft', 'left', 'middle', 'right', 'rright']
    velocities = args.velocities #[0, 10, 20]

    # Init environment and buffer
    env = CarRacing()
    buffer = ReplayBuffer.create_empty_numpy()

    # ======================  GENERATE DATA  ====================== #
    for mode in modes:
        print("Mode: ", mode)
        img_hist, vel_hist ,act_hist, track_hist = driving(env, buffer, NUM_EPISODES_PER_MODE, mode, velocities)

    # ======================  SAVE DATA  ====================== #
    today = datetime.now()    # Get today's date
    folder_name = today.strftime("%Y-%m-%d-%H%M")     # Format the date as a string
    # Check if the folder exists
    if not os.path.exists(base_dir + folder_name):
        # Create the folder
        os.makedirs(base_dir + folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists.")

    if dataset_name:
        dataset_name = dataset_name.rstrip('.zarr')
        dataset_name = dataset_name + f"_dataset_{NUM_EPISODES_PER_MODE}_episodes_{len(modes)}_modes.zarr"
    else:
        dataset_name = folder_name + f"_dataset_{NUM_EPISODES_PER_MODE}_episodes_{len(modes)}_modes.zarr"
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
    parser.add_argument("--num_episodes_per_mode", type=int, default=1, help="Number of episodes per mode")
    parser.add_argument("--chunk_len", type=int, default=-1, help="Chunk length")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name")
    parser.add_argument("--base_dir", type=str, default="./data/", help="Base directory")
    parser.add_argument("--modes", nargs="+", default=["middle", "left", "right"], help="Modes list")
    parser.add_argument("--velocities", nargs="+", default=[  20 ], help="Velocities list")
    parser.add_argument("--render_mode", type=str, default="human", help="render mode of gym env. human means render, rgb_array means no render visible")

    args = parser.parse_args()

    render_mode = args.render_mode

    print()
    print("======== Parameters =======")
    print("Number of Episodes per Mode:", args.num_episodes_per_mode)
    print("Chunk Length:", args.chunk_len)
    print("Dataset Name:", args.dataset_name)
    print("Base Directory:", args.base_dir)
    print("Modes:", args.modes)
    print("Velocities:", args.velocities)
    print("Render Mode:", args.render_mode)
    print("============================")
    print()

    generateData(args)
