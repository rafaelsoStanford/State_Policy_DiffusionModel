import sys
# setting path
sys.path.append('../diffusion_bare')

import numpy as np
from collections import deque
from simple_pid import PID
from datetime import datetime

from utils.replay_buffer import ReplayBuffer
from envs.car_racing import CarRacing
from generateData.trajectory_control_utils import *

# -----  Initialize buffers ----- #
buffer = ReplayBuffer.create_empty_numpy()

error_velocity_buffer =     deque(np.zeros(7), maxlen = 7)
error_buffer =              deque(np.zeros(10), maxlen = 10) 
error_buffer_2 =            deque(np.zeros(3), maxlen = 3) 
# -----  Initialize history lists ----- #
img_hist, vel_hist ,act_hist, pos_hist, angle_hist = [], [], [], [], []


#----- PID controllers ----- #
pid_velocity = PID(0.005, 0.001, 0.0005, setpoint=20)
pid_steering = PID(0.8, 0.01, 0.06, setpoint=0) 

# -----  Initialize environment ----- #
env = CarRacing()
env.seed(42)
env.reset()
action = np.array([0, 0, 0], dtype=np.float32)
obs, _ , _, info = env.step(action) # Take a step to get the environment initialized (action is empty)

modes = "left" # ["left", "right", "middle"]

# ======================  START EPISODE  ====================== #
for _ in range(2000):
    env.render("human")
    augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
    
    # Calculate action
    action_ = trajectory_control(augmImg, pid_steering, pid_velocity, 
                error_buffer, error_buffer_2, error_velocity_buffer, v_wFrame, modes)

    if action_ is None: # In the tightest curve we lose intersection of strip with trajectory
        obs, _ , _ , info = env.step(action)
        augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)          
        append_to_histories(obs, carVelocity_wFrame, carPosition_wFrame, action, car_heading_angle, img_hist, vel_hist, pos_hist, act_hist, angle_hist)
        continue

    action = action_
    obs, _ , _ , info = env.step(action)      
    append_to_histories(obs, carVelocity_wFrame, carPosition_wFrame, action, car_heading_angle, img_hist, vel_hist, pos_hist, act_hist, angle_hist)
env.close()
# ======================  END EPISODE  ====================== #

# ---- Saving the trajectory ---- #
dataset_name = "EvaluationDataset_" + modes
base_dir = "./evaluation/data"
today = datetime.now()
folder_name = today.strftime("%Y-%m-%d-%H%M")

save_data_to_buffer(buffer, img_hist, act_hist, vel_hist, pos_hist, angle_hist)
create_folder(base_dir, folder_name)
dataset_name = generate_dataset_name(dataset_name, folder_name, 1, 1)
save_buffer_to_zarr(buffer, base_dir, folder_name, dataset_name, -1)

  


