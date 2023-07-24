

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
from models.diffusion_ddpm import *


def fetch_hyperparams_from_yaml(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams


def normalize_position(sample, stats):
    sample_normalized = normalize_data(sample, stats['position'])
    translation_vec = sample_normalized[0, :]
    nsample_centered = sample_normalized - translation_vec
    nsample = nsample_centered / 2.0
    return nsample, translation_vec


# =========== Load data ===========
dataset_path = "./evaluation/data/2023-07-21-1537/EvaluationDataset_left_dataset_1_episodes_1_modes.zarr.zip"
filepath = './tb_logs/version_624/STATS.pkl'
path_checkpoint = './tb_logs/version_624/checkpoints/epoch=35.ckpt'
path_hyperparams = './tb_logs/version_624/hparams.yaml'

print("*** Loading Data ...")
dataset_root = zarr.open(dataset_path, 'r')
data = {
    'image': np.moveaxis(dataset_root['data']['img'][:], -1, 1),
    'position': dataset_root['data']['position'][:],
    'velocity': dataset_root['data']['velocity'][:],
    'action': dataset_root['data']['action'][:],
}
model_params = fetch_hyperparams_from_yaml(path_hyperparams)
obs_horizon = model_params['obs_horizon']
pred_horizon = model_params['pred_horizon']
inpaint_horizon = model_params['inpaint_horizon']
print("     ... Data Loaded ***")

print("*** Loading Stats ...")
# Load the pickle file
with open(filepath, 'rb') as f:
    stats = pickle.load(f)[0]
print("     ... Stats Loaded ***")

print("*** Generate indices of possible trajectory samples ***")
sequence_len = obs_horizon + pred_horizon
indices = create_sample_indices(
    episode_ends=[len(data['position'])],
    sequence_length=sequence_len,
    pad_before=0,
    pad_after=0
)
print("*** Randomly chosen trajectory sample ...")
buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx =  indices[np.random.randint(0, len(indices))]
sample = sample_sequence(
    train_data= data,
    sequence_length= sequence_len,
    buffer_start_idx= buffer_start_idx,
    buffer_end_idx= buffer_end_idx,
    sample_start_idx= sample_start_idx,
    sample_end_idx= sample_end_idx
)
start_t = buffer_start_idx
end_t = buffer_end_idx
print("     ... Trajectory sample obtained ***")
print("-" * 10 )
print(" -- Lenght of sequence: ", len(sample['position']))
print(" -- Buffer start index: ", buffer_start_idx)
print(" -- Buffer end index: ", buffer_end_idx)
print("-" * 10 )


print( "*** Normalize sample ...")
nAction = normalize_data(sample['action'], stats=stats['action'])
nVelocity = normalize_data(sample['velocity'], stats=stats['velocity'])
nPosition, Translation = normalize_position(sample['position'], stats)

nsample = {
    'position': nPosition,
    'velocity': nVelocity,
    'action': nAction,
    'image': sample['image'], # already normalized
}
print("     ... Sample normalized ***")


# -----  Loading and initializing model ----- #
print("*** Loading Model ...")
model = Diffusion_DDPM.load_from_checkpoint(
    path_checkpoint,
    hparams_file=path_hyperparams,
)
model.eval()
print("     ... Model Loaded ***")


# # -----  Prepare observation vector ----- #
print("*** Preparing observation vector ...")
obs_position = nsample['position'][:obs_horizon]
obs_velocity = nsample['velocity'][:obs_horizon]
obs_action = nsample['action'][:obs_horizon]
obs_image = nsample['image'][:obs_horizon]

print(obs_position.shape)
print(obs_velocity.shape)
print(obs_action.shape)
print(obs_image.shape)

print("Min/Max position: ", np.min(obs_position), np.max(obs_position))
print("Min/Max velocity: ", np.min(obs_velocity), np.max(obs_velocity))
print("Min/Max action: ", np.min(obs_action), np.max(obs_action))
print("Min/Max image: ", np.min(obs_image), np.max(obs_image))

print("... Observation vector prepared ***")


print("*** Create Batch ... ")
obs_position = torch.from_numpy(obs_position).unsqueeze(0).float()
obs_velocity = torch.from_numpy(obs_velocity).unsqueeze(0).float()
obs_action = torch.from_numpy(obs_action).unsqueeze(0).float()
obs_image = torch.from_numpy(obs_image).unsqueeze(0).float()

print(obs_position.shape)
print(obs_velocity.shape)
print(obs_action.shape)
print(obs_image.shape)

batch = {
    'position': torch.from_numpy(nsample['position']).unsqueeze(0).float(),
    'velocity': torch.from_numpy(nsample['velocity']).unsqueeze(0).float(),
    'action': torch.from_numpy(nsample['action']).unsqueeze(0).float(),
    'image': torch.from_numpy(nsample['image']).unsqueeze(0).float(),
}

for key, value in batch.items():
    print(key, value.shape)
print("... Batch created ***")


print("*** Sample from model ***")
x_0_predicted, _, _ = model.sample(batch=batch, mode='validation')
print("      ... Sampled ***")

print("*** Denormalize sample ***")
x_0_predicted = x_0_predicted.squeeze().cpu().detach().numpy()
nAction_predicted = x_0_predicted[:, 2:5]
action_pred = unnormalize_data(nAction_predicted, stats['action']) # Includes inpainted points
nPosition_predicted = x_0_predicted[:, 0:2]
position_pred = unnormalize_data(2* nPosition_predicted + Translation, stats['position']) # Includes inpainted points
action_pred = action_pred[inpaint_horizon:]
position_pred = position_pred[inpaint_horizon:]

print(" Sampled action: ", action_pred.shape)
print(" Sampled position: ", position_pred.shape)
print("*** Denormalized ***")

# -----  Initialize history lists ----- #
img_hist, vel_hist ,act_hist, pos_hist, angle_hist = [], [], [], [], []
# -----  Initialize environment ----- #
env = CarRacing()
env.seed(42)
env.reset()
action = np.array([0, 0, 0], dtype=np.float32)
obs, _ , _, info = env.step(action) # Take a step to get the environment initialized (action is empty)

# ======================  START EPISODE  ====================== #
for i in range(start_t + obs_horizon):
    env.render('state_pixels')
    augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
    obs, _ , _, info = env.step(data['action'][i]) 
    # append_to_histories(obs, carVelocity_wFrame, carPosition_wFrame, action, car_heading_angle, img_hist, vel_hist, pos_hist, act_hist, angle_hist)


for i in range(pred_horizon):
    env.render('human')
    augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
    obs, _ , _, info = env.step(action_pred[i]) 
    append_to_histories(obs, carVelocity_wFrame, carPosition_wFrame, action, car_heading_angle, img_hist, vel_hist, pos_hist, act_hist, angle_hist)
pos_hist = np.array(pos_hist)
actions_saved_traj = pos_hist
env.close()


# ? Sanity check : Comparison to true actions
# -----  Initialize history lists ----- #
img_hist, vel_hist ,act_hist, pos_hist, angle_hist = [], [], [], [], []
# -----  Initialize environment ----- #
env = CarRacing()
env.seed(42)
env.reset()
action = np.array([0, 0, 0], dtype=np.float32)
obs, _ , _, info = env.step(action) # Take a step to get the environment initialized (action is empty)
for i in range(end_t):
    env.render('state_pixels')
    augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
    obs, _ , _, info = env.step(data['action'][i]) 
    append_to_histories(obs, carVelocity_wFrame, carPosition_wFrame, action, car_heading_angle, img_hist, vel_hist, pos_hist, act_hist, angle_hist)
pos_hist = np.array(pos_hist)
observed_traj = pos_hist[start_t:]



plt.plot(data['position'][:,0],data['position'][:,1], c='b', label = "Track Groundtruth positions")
plt.scatter(observed_traj[:,0], observed_traj[:,1], c='c', s=30, marker='o' , label = "Observed trajectory from groundtruth actions")
plt.scatter(data['position'][start_t:end_t,0], data['position'][start_t:end_t,1], c='r', s=20, marker='o', label = "Groundtruth trajectory from groundtruth positions")
plt.scatter(actions_saved_traj[:,0], actions_saved_traj[:,1], c='y', s=20, marker='x', label = "Diffusion based actions observed trajectory")
plt.scatter(position_pred[:,0], position_pred[:,1], c='g', s=20, marker='x', label = "Diffusion based trajectory")
plt.legend()
plt.show()


