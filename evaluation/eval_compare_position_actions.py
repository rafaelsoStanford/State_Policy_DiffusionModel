import sys
import pickle
import yaml
import torch
import numpy as np
import zarr
import matplotlib.pyplot as plt

# setting path
sys.path.append('../diffusion_bare')

from envs.car_racing import CarRacing
from models.diffusion_ddpm import *
from models.diffusion_ddim import *
from utils.load_data import *
from utils.data_utils import *
from generateData.trajectory_control_utils import *


# ###### GLOBAL VARIABLES ######
NUM_RUNS = 10
MODE =  'state_pixels'
ENV_SEED = 42
EXPERIMENT_NAME = 'Consistency evaluations'
MODEL_NAME = 'DDIM'

def interpolate_actions(current_action, next_action, steps):
    """Linearly interpolate between current and next action for a number of steps."""
    return np.array([np.linspace(current, next, steps) for current, next in zip(current_action, next_action)]).T

# # paths
dataset_path = './evaluation/data/'
dataset_name = 'Evaluation_dataset_1_episodes_1_modes.zarr.zip'

filepath = './tb_logs/version_671/STATS.pkl'
path_checkpoint = './tb_logs/version_671/checkpoints/epoch=10.ckpt'
path_hyperparams = './tb_logs/version_671/hparams.yaml'


print("*** Loading Data ...")
dataset_root = zarr.open(dataset_path + dataset_name, 'r')
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
step_size = model_params['step_size']

print("*** Loading Stats ...")
with open(filepath, 'rb') as f:
    stats = pickle.load(f)[0]
print("     ... Stats Loaded ***")


print("*** Loading Model ...")
# -----  Loading and initializing model ----- #
model = Diffusion_DDPM.load_from_checkpoint(
    path_checkpoint,
    hparams_file=path_hyperparams,
)
model.eval()

# model = Diffusion_DDIM.load_from_checkpoint(
#     path_checkpoint,
#     hparams_file=path_hyperparams,
# )
print("     ... Model Loaded ***")


print("*** Generate indices of possible trajectory samples ***")
sequence_len = obs_horizon + pred_horizon
indices = create_sample_indices_sparse(
    ends= [data['position'].shape[0]],
    sequence_length= sequence_len,
    step_size= step_size)


print("*** Randomly chosen trajectory sample ...")
start_idx, end_idx, _, _ =  indices[70]  # indices[np.random.randint(0, len(indices))]
sample = sample_sequence_sparse(
    data= data,
    step_size= step_size,
    sample_start_idx= start_idx,
    sample_end_idx= end_idx,
)

print("     ... Trajectory sample obtained ***")


print( "*** Normalize sample ...")
nAction = normalize_data(sample['action'], stats=stats['action'])
nVelocity = normalize_data(sample['velocity'], stats=stats['velocity'])
nPosition, Translation = normalize_position(sample['position'], stats['position'])
# Construct normalized sample dict
nsample = {
    'position': nPosition,
    'velocity': nVelocity,
    'action': nAction,
    'image': sample['image'], # already normalized
}
print("     ... Sample normalized ***")


batch = {
    'position': torch.from_numpy(nsample['position']).unsqueeze(0).float(),
    'velocity': torch.from_numpy(nsample['velocity']).unsqueeze(0).float(),
    'action': torch.from_numpy(nsample['action']).unsqueeze(0).float(),
    'image': torch.from_numpy(nsample['image']).unsqueeze(0).float(),
}
print("     ... Batch created ***")

print("Parameter overview:")
print("Obs horizon: ", obs_horizon)
print("Pred horizon: ", pred_horizon)
print("Inpaint horizon: ", inpaint_horizon)
print("Step size: ", step_size)

print( " In seconds: ")
print("Obs horizon: ", obs_horizon * step_size* 1/50)
print("Pred horizon: ", pred_horizon * step_size* 1/50)
print("Inpaint horizon: ", inpaint_horizon * step_size* 1/50)
print("Step size: ", step_size* 1/50)


# # ================================================================================= #
# # ======================  Run N runs of the same experiment  ====================== #
# # ================================================================================= #

trajectory_list = []
pos_pred_list = []
actions_list = []
for i in range(NUM_RUNS):
    print()
    print("*** Sample from model ***")
    x_0_predicted, _, _ = model.sample(batch=batch, mode='validation')
    # x_0_predicted, _, _ = model.sample(batch=batch, mode='validation', step_size=10 , ddpm_steps=100)
    print("      ... Sampled ***")

    print("*** Unnormalize sample ***")
    x_0_predicted = x_0_predicted.squeeze().cpu().detach().numpy()
    nAction_predicted = x_0_predicted[:, 2:5]
    action_pred = unnormalize_data(nAction_predicted, stats['action']) # Includes inpainted points
    action_pred = action_pred[inpaint_horizon:]

    nPosition_predicted = x_0_predicted[:, 0:2]
    position_pred = unnormalize_position(nPosition_predicted , Translation , stats['position']) 
    pos_pred_list.append(position_pred.copy())
    print("     ...Unnormalized ***")

    # -----  Initialize environment ----- #
    env = CarRacing()
    env.seed(ENV_SEED)
    env.reset()
    action = np.array([0, 0, 0], dtype=np.float32)
    obs, _ , _, info = env.step(action) # Take a step to get the environment initialized (action is empty)
    # -----  Initialize history lists ----- #
    pos_hist = []
    for i in range(start_idx + obs_horizon * step_size):
        env.render(MODE)
        augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
        obs, _ , _, info = env.step(data['action'][i])
    current_action = data['action'][start_idx + obs_horizon - 1]
    
    # Run Predicted actions
    action_hist = []
    for i in range(pred_horizon * step_size):
        env.render(MODE)
        augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
        if i % step_size == 0:
            if i < pred_horizon * step_size - step_size:  # Ensure next_action exists
                next_action = action_pred[i // step_size + 1]
                interpolated_actions = interpolate_actions(current_action, next_action, step_size)
            obs, _ , _, info = env.step(interpolated_actions[i % step_size])
            current_action = action_pred[i // step_size]
        else:
            obs, _ , _, info = env.step(interpolated_actions[i % step_size])

        pos_hist.append(carPosition_wFrame.copy())
        action_hist.append(interpolated_actions[i % step_size])

    trajectory_list.append(pos_hist)
    action_hist = np.array(action_hist)
    actions_list.append(action_hist)

    print("     ... Trajectory generated ***")
    env.close()


# # ================================================================================= #
# # ================================================================================= #


# Calculate error between predicted trajectory and action rollout using Frechet Distance

def euclidean_dist(pt1, pt2):
    return np.sqrt(np.sum((pt1-pt2)**2))

def plot_pointwise_distances(trajectory1_list, trajectory2_list, NUM_RUNS):
    """
    Calculates the pointwise Euclidean distances between two trajectories and plots them.
    """
    all_distances = []

    for run in range(NUM_RUNS):
        trajectory1 = trajectory1_list[run][::step_size]
        trajectory2 = trajectory2_list[run][inpaint_horizon:,:]

        distances = [euclidean_dist(pt1, pt2) for pt1, pt2 in zip(trajectory1, trajectory2)]
        all_distances.append(distances)

    # Convert list of lists to numpy array for easier manipulation
    all_distances = np.array(all_distances)

    # # Plot all distances
    # plt.figure(figsize=(10, 6))
    # for run in range(NUM_RUNS):
    #     plt.plot(all_distances[run], alpha=0.5)
    # plt.xlabel('Index')
    # plt.ylabel('Euclidean Distance')
    # plt.title(f'Pointwise Euclidean Distances Between Two Trajectories Over {NUM_RUNS} Runs\n'
    #       f'Observation Horizon: {obs_horizon * step_size * 1/50} seconds, Prediction Horizon: {pred_horizon * step_size * 1/50} seconds')
    # plt.grid(True)
    # plt.show()

    # # Plot mean distances
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.mean(all_distances, axis=0))
    # plt.xlabel('Index')
    # plt.ylabel('Mean Euclidean Distance')
    # plt.title(f'Mean Pointwise Euclidean Distances Between Two Trajectories Over {NUM_RUNS} Runs\n'
    #       f'Observation Horizon: {obs_horizon * step_size * 1/50} seconds, Prediction Horizon: {pred_horizon * step_size * 1/50} seconds')
    # plt.grid(True)
    # plt.show()
    # # Plot mean distances
    # plt.figure(figsize=(10, 6))
    # for run in range(NUM_RUNS):
    #     plt.plot(all_distances[run], alpha=0.5, color='blue', label='Single Run' if run == 0 else "")
    # plt.plot(np.mean(all_distances, axis=0), linewidth=2, color='red', label='Mean Average Over All Runs')
    # plt.xlabel('Index')
    # plt.ylabel('Mean Euclidean Distance')
    # plt.title(f'Mean Pointwise Euclidean Distances Between Two Trajectories Over {NUM_RUNS} Runs\n'
    #     f'Observation Horizon: {obs_horizon * step_size * 1/50} seconds, Prediction Horizon: {pred_horizon * step_size * 1/50} seconds')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # Plot mean distances
    plt.figure(figsize=(10, 6))
    for run in range(NUM_RUNS):
        plt.plot(np.array(range(len(all_distances[run]))) / 50 * step_size, all_distances[run], alpha=0.5, color='blue', label='Single Run' if run == 0 else "")
    plt.plot(np.array(range(len(all_distances[0]))) / 50 * step_size, np.mean(all_distances, axis=0), linewidth=2, color='red', label='Mean Average Over All Runs')
    plt.fill_between(np.array(range(len(all_distances[0]))) / 50 * step_size, np.mean(all_distances, axis=0) - np.std(all_distances, axis=0), 
                    np.mean(all_distances, axis=0) + np.std(all_distances, axis=0), color='red', alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mean Euclidean Distance')
    plt.title(f'Mean Pointwise Euclidean Distances Between Two Trajectories Over {NUM_RUNS} Runs\n'
        f'Observation Horizon: {obs_horizon * step_size * 1/50} seconds, Prediction Horizon: {pred_horizon * step_size * 1/50} seconds')
    plt.grid(True)
    plt.legend()
    plt.show()



# Assuming trajectory_list and pos_pred_list are lists of lists, where each inner list is a trajectory
plot_pointwise_distances(np.array(trajectory_list), np.array(pos_pred_list), NUM_RUNS)
