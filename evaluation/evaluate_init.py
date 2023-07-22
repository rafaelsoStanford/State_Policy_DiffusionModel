

import matplotlib.pyplot as plt
import yaml
import sys
import zarr
import numpy as np
from collections import deque
from simple_pid import PID

# setting path
sys.path.append('../diffusion_bare')

from envs.car_racing import CarRacing
from utils.load_data import *
from generateData.trajectory_control_utils import *
from utils.replay_buffer import ReplayBuffer


def fetch_hyperparams_from_yaml(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams


# =========== Load data ===========
# path_hyperparams = './tb_logs/version_624/hparams.yaml'
# path_checkpoint = './tb_logs/version_624/checkpoints/epoch=35.ckpt'
# filepath = './tb_logs/version_624/STATS.pkl'
dataset_path = "./evaluation/data/2023-07-21-1537/EvaluationDataset_left_dataset_1_episodes_1_modes.zarr.zip"

print("*** Loading Data ***")
dataset_root = zarr.open(dataset_path, 'r')
train_data = {
    'position': dataset_root['data']['position'][:],
    'action': dataset_root['data']['action'][:],
}
print("*** Data Loaded ***")

sequence_len = 100
indices = create_sample_indices(
    episode_ends=[len(train_data['position'])],
    sequence_length=sequence_len,
    pad_before=0,
    pad_after=0
)

print(indices.shape)
print(type(indices))

buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx =  indices[np.random.randint(0, len(indices))]

print("buffer_start_idx: ", buffer_start_idx)
print("buffer_end_idx: ", buffer_end_idx)
print("sample_start_idx: ", sample_start_idx)
print("sample_end_idx: ", sample_end_idx)

nsample = sample_sequence(
    train_data= train_data,
    sequence_length= sequence_len,
    buffer_start_idx= buffer_start_idx,
    buffer_end_idx= buffer_end_idx,
    sample_start_idx= sample_start_idx,
    sample_end_idx= sample_end_idx
)

print(nsample['position'].shape)
print(nsample['action'].shape)
start_t = buffer_start_idx

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

modes = "left" 
# ======================  START EPISODE  ====================== #
for _ in range(start_t):
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


img_hist_2, vel_hist_2 ,act_hist_2, pos_hist_2, angle_hist_2 = [], [], [], [], []
for _ in range(sample_end_idx):
    env.render("human")
    augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
    # Calculate action
    action_ = trajectory_control(augmImg, pid_steering, pid_velocity,
                error_buffer, error_buffer_2, error_velocity_buffer, v_wFrame, modes)

    if action_ is None: # In the tightest curve we lose intersection of strip with trajectory
        obs, _ , _ , info = env.step(action)
        augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)          
        append_to_histories(obs, carVelocity_wFrame, carPosition_wFrame, action, car_heading_angle, img_hist_2, vel_hist_2, pos_hist_2, act_hist_2, angle_hist_2)
        continue

    action = action_
    obs, _ , _ , info = env.step(action)      
    append_to_histories(obs ,carVelocity_wFrame, carPosition_wFrame, action, car_heading_angle, img_hist_2, vel_hist_2, pos_hist_2, act_hist_2, angle_hist_2)
env.close()
# ======================  END EPISODE  ====================== #


# plt.plot(train_data['position'][:,0],train_data['position'][:,1], c='b')
plt.scatter(train_data['position'][:,0],train_data['position'][:,1], c='b', s=10, marker='x')
plt.legend(['Track left'])
plt.plot(train_data['position'][start_t:start_t+sequence_len,0],train_data['position'][start_t:start_t+sequence_len,1])
plt.legend(['Track starting at buffer_start_idx'])
pos_hist = np.array(pos_hist)
plt.scatter(pos_hist[:,0], pos_hist[:,1], c='r', s=10, marker='x')
pos_hist_2 = np.array(pos_hist_2)
plt.scatter(pos_hist_2[:,0], pos_hist_2[:,1], c='g', s=10, marker='x')
plt.show()
plt.close()
