import sys

import numpy as np
import argparse
from simple_pid import PID 
from collections import deque
from datetime import datetime

import pickle

# setting path
sys.path.append('../diffusion_bare')

from envs.car_racing import CarRacing
from utils.replay_buffer import ReplayBuffer
import trajectory_control_utils as tc_utils

# -------Global variables------- #
global render_mode 
max_steps = 2000
seeds =  [42] # np.random.randint(43, 500, size=NUM_EPISODES) # [42] #


def driving(buffer, NUM_EPISODES, MODE, VELOCITIES):
    # ======================  START RUN  ====================== #
    print("*"*10 +" Starting run...: Current Mode: ", MODE, "*"*10)
    for episode in range(NUM_EPISODES):   
        # -----  Initialize buffers ----- #
        error_velocity_buffer =     deque(np.zeros(7), maxlen = 7)
        error_buffer =              deque(np.zeros(10), maxlen = 10) 
        error_buffer_2 =            deque(np.zeros(3), maxlen = 3) 
        # -----  Initialize history lists ----- #
        img_hist, vel_hist ,act_hist, pos_hist, angle_hist = [], [], [], [], []

        # State history
        state_hist = []

        #----- PID controllers ----- #
        pid_velocity = PID(0.005, 0.001, 0.0005, setpoint=VELOCITIES[0])
        pid_steering = PID(0.8, 0.01, 0.06, setpoint=0) 
        
        # -----  Initialize environment ----- #
        env = CarRacing()
        env.seed(int(seeds[episode]))
        env.reset()
        action = np.array([0, 0, 0], dtype=np.float32)
        obs, _ , _, info = env.step(action) # Take a step to get the environment initialized (action is empty)

        # ======================  START EPISODE  ====================== #
        for _ in range(max_steps):
            env.render(render_mode)
            augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = tc_utils.extract_variables(info)
            
            # Calculate action
            action_ = tc_utils.trajectory_control(augmImg, pid_steering, pid_velocity, 
                       error_buffer, error_buffer_2, error_velocity_buffer, v_wFrame, MODE)

            if action_ is None: # In the tightest curve we lose intersection of strip with trajectory
                obs, _ , _ , info = env.step(action)
                augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = tc_utils.extract_variables(info)          
                tc_utils.append_to_histories(obs, carVelocity_wFrame, carPosition_wFrame, action, car_heading_angle, img_hist, vel_hist, pos_hist, act_hist, angle_hist)
                continue

            action = action_
            obs, _ , _ , info = env.step(action)      
            tc_utils.append_to_histories(obs, carVelocity_wFrame, carPosition_wFrame, action, car_heading_angle, img_hist, vel_hist, pos_hist, act_hist, angle_hist)

            state = env.car._save_state()
            state_hist.append(state)

        env.close()

        # -----  Save state history ----- #
        # Save to a binary file
        with open("states_list.pkl", "wb") as file:
            pickle.dump(state_hist, file)
        
        # -----  Save data to buffer ----- #
        tc_utils.save_data_to_buffer(buffer, img_hist, act_hist, vel_hist, pos_hist, angle_hist)
    return img_hist, vel_hist ,act_hist, pos_hist # For debugging purposes


def generate_data(args):
    num_episodes_per_mode = args.num_episodes_per_mode
    chunk_len = args.chunk_len
    dataset_name = args.dataset_name
    base_dir = args.base_dir
    modes = args.modes
    velocities = args.velocities
    buffer = ReplayBuffer.create_empty_numpy()

    for mode in modes:
        driving(buffer, num_episodes_per_mode, mode, velocities)

    today = datetime.now()
    folder_name = today.strftime("%Y-%m-%d-%H%M")

    tc_utils.create_folder(base_dir, folder_name)
    dataset_name = tc_utils.generate_dataset_name(dataset_name, folder_name, num_episodes_per_mode, len(modes))
    tc_utils.save_buffer_to_zarr(buffer, base_dir, folder_name, dataset_name, chunk_len)


# ======================  MAIN  ====================== #
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate Data")
    parser.add_argument("--num_episodes_per_mode", type=int, default=1, help="Number of episodes per mode")
    parser.add_argument("--chunk_len", type=int, default=-1, help="Chunk length")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name")
    parser.add_argument("--base_dir", type=str, default="./data/", help="Base directory")
    parser.add_argument("--modes", nargs="+", default=["left" , "right"], help="Modes list")
    parser.add_argument("--velocities", nargs="+", default=[ 30 ], help="Velocities list")
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

    generate_data(args)
