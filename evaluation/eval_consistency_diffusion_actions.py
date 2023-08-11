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


# # ==================================================== #
# # Plot sample
# print("*** Plotting sample ...")
# fig, axs = plt.subplots()
# plt.plot(nPosition[:, 0], nPosition[:, 1])
# plt.title('Position')
# axs.set_aspect('equal', 'box')
# plt.show()
# # ==================================================== #

# # ================================================================================= #
# # ======================  Run N runs of the same experiment  ====================== #
# # ================================================================================= #

trajectory_list = []
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

    #Run Predicted actions zero hold
    # current_action = data['action'][start_idx + obs_horizon - 1]
    # for i in range(pred_horizon * step_size):
    #     env.render(MODE)
    #     augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
    #     if i % step_size == 0:
    #         obs, _ , _, info = env.step(action_pred[i//step_size])
    #         #current_action = action_pred[i//step_size]
    #     else:
    #         obs, _ , _, info = env.step(np.array([0,0,0]))   # current_action)
    #     pos_hist.append(carPosition_wFrame)

    print("     ... Trajectory generated ***")
    env.close()

# # ================================================================================= #
fig, axs = plt.subplots(figsize=(10, 10))
plt.scatter(data['position'][start_idx:start_idx +(obs_horizon*step_size):step_size, 0], data['position'][start_idx:start_idx +(obs_horizon*step_size):step_size, 1], label='Observed position points', c='r', s=30)

flag = False
for i in range(len(trajectory_list)):
    if flag is False:
        plt.plot(np.array(trajectory_list[i])[::step_size, 0], np.array(trajectory_list[i])[::step_size, 1], color='r', alpha=0.5)
        plt.scatter(np.array(trajectory_list[i])[::step_size, 0], np.array(trajectory_list[i])[::step_size, 1], color='r', alpha=0.5, s=10, label='Predicted trajectories')
        flag = True
    else:
        plt.plot(np.array(trajectory_list[i])[::step_size, 0], np.array(trajectory_list[i])[::step_size, 1], color='r', alpha=0.5)
        plt.scatter(np.array(trajectory_list[i])[::step_size, 0], np.array(trajectory_list[i])[::step_size, 1], color='r', alpha=0.5, s=10)

plt.plot(data['position'][start_idx:end_idx:step_size, 0], data['position'][start_idx:end_idx:step_size, 1], label='Ground Truth', c='g', linewidth=2)
plt.scatter(data['position'][start_idx:end_idx:step_size, 0], data['position'][start_idx:end_idx:step_size, 1], c='g', s=20, label='Ground Truth points')

plt.legend()
axs.set_aspect('equal', 'box')

# Title with model parameters included
plt.title(f'Comparison of Ground Truth Trajectory and Predicted Rollouts\n'
          f'Obs Horizon: {obs_horizon}, Pred Horizon: {pred_horizon}\n'
          f'Inpaint Horizon: {inpaint_horizon}, Step Size: {step_size}')

plt.axis('scaled')
plt.show()



## ================================================================================= #
# Plotting predicted actions and ground truth actions
fig, axs = plt.subplots(figsize=(10, 10))
# Creating time vector
time_vector = np.arange(start_idx + inpaint_horizon*step_size, start_idx + ((obs_horizon+pred_horizon) * step_size), step_size) / 50.0  # divide by FPS to convert to seconds

# Emphasizing ground truth by using a larger line width and bold color
plt.plot(time_vector, data['action'][start_idx + inpaint_horizon*step_size :start_idx + ((obs_horizon+pred_horizon) * step_size):step_size, 0], label='Ground Truth', c='g', linewidth=2, marker='o')

flag = False
for i in range(len(actions_list)):
    action_array = np.array(actions_list[i])[::step_size,0]
    # Plot predicted action signals in red, with a single label for all of them
    if flag is False:
        plt.plot(time_vector[:len(action_array)], action_array, color='r', alpha=0.5, marker='o', label='Predicted')
        flag = True
    else:
        plt.plot(time_vector[:len(action_array)], action_array, color='r', alpha=0.5, marker='o')

plt.legend()
plt.xlabel('Time (s)')  # labeling x-axis as time
plt.title(f'Comparison of Ground Truth Action and Predicted Actions\n'
          f'Obs Horizon: {obs_horizon}, Pred Horizon: {pred_horizon}\n'
          f'Inpaint Horizon: {inpaint_horizon}, Step Size: {step_size}')
plt.show()


# Initialize a list to store all errors for averaging
all_errors = []

# Initialize a figure for the error plot
fig_error, ax_error = plt.subplots()

# Loop over each predicted position trajectory
for i in range(len(trajectory_list)):
    # Calculate the Euclidean distance between the predicted positions and the ground truth positions
    error = np.sqrt(np.sum((np.array(trajectory_list[i])[::step_size, :2] - data['position'][start_idx + inpaint_horizon*step_size :start_idx + ((obs_horizon+pred_horizon) * step_size):step_size, :2])**2, axis=1))
    # Store the error
    all_errors.append(error)
    # Plot the error over time in red and slightly transparent, only label the first one
    if i == 0:
        ax_error.plot(time_vector[:len(error)], error, color='r', alpha=0.5, label=f'Run 1 to {len(trajectory_list)}')
    else:
        ax_error.plot(time_vector[:len(error)], error, color='r', alpha=0.5)

# Calculate the average error
average_error = np.mean(np.array(all_errors), axis=0)
# Calculate the standard deviation
std_dev = np.std(np.array(all_errors), axis=0)

# Plot the average error over time in blue, bold
ax_error.plot(time_vector[:len(average_error)], average_error, color='b', linewidth=2, label='Average Error')

# Fill the region within one standard deviation of the average error
ax_error.fill_between(time_vector[:len(average_error)], average_error - std_dev, average_error + std_dev, color='b', alpha=0.1)

# Formatting the plot
ax_error.set_xlabel('Time (s)')
ax_error.set_ylabel('Error (Euclidean Distance)')
ax_error.set_title(f'Average Error of Predicted Action Rollout Sequences over Time compared to ground truth\n'
                    f'Obs Horizon: {obs_horizon}, Pred Horizon: {pred_horizon}, Step Size: {step_size}')
ax_error.legend()

plt.show()




# # Plotting Positions
# fig, axs = plt.subplots()
# plt.scatter(data['position'][start_idx:start_idx +(obs_horizon*step_size):step_size, 0], data['position'][start_idx:start_idx +(obs_horizon*step_size):step_size, 1], label='Observed position points', c='r', s=30)
# for i in range(len(trajectory_list)):
#     # Plot rollout trajecotries
#     plt.plot(np.array(trajectory_list[i])[::step_size, 0], np.array(trajectory_list[i])[::step_size, 1], label='Predicted trajectory ' + str(i))
#     plt.scatter(np.array(trajectory_list[i])[::step_size, 0], np.array(trajectory_list[i])[::step_size, 1], s=10)

# plt.plot(data['position'][start_idx:end_idx:step_size, 0], data['position'][start_idx:end_idx:step_size, 1], label='Ground Truth')
# plt.scatter(data['position'][start_idx:end_idx:step_size, 0], data['position'][start_idx:end_idx:step_size, 1], c='g', s=10, label='Ground Truth')
# plt.legend()
# axs.set_aspect('equal', 'box')
# plt.title('Position')
# plt.show()

# # ================================================================================= #
# prediction_actions = unnormalize_data(nAction_predicted, stats['action']) 
# x_axis = np.arange(start_idx, end_idx, step_size)

# plt.figure()
# plt.plot(x_axis, data['action'][start_idx:end_idx:step_size, 0], c='b', label='Ground Truth')

# # Create x_axis for prediction_actions, assuming it starts at start_idx
# pred_x_axis = np.arange(start_idx , start_idx + pred_horizon * step_size + inpaint_horizon*step_size, step_size)
# plt.plot(pred_x_axis, prediction_actions, c='r', label='Predicted')

# plt.title('Steering')
# plt.legend()
# plt.show()





# #Model
# model = load_model(MODEL_NAME,  path_checkpoint, path_hyperparams)
# model_params = fetch_hyperparams_from_yaml(path_hyperparams)
# obs_horizon = model_params['obs_horizon']
# pred_horizon = model_params['pred_horizon']

# # ==================================================== #
# # Print hyperparameter information
# print('Hyperparameters:')
# print('Model: ', MODEL_NAME)
# print('Step size: ', step_size , 'equivalent to ', step_size*1/50, 'seconds')
# print('Obs horizon: ', obs_horizon , " equivalent to ", step_size*obs_horizon*1/50, "seconds")
# print('Pred horizon: ', pred_horizon , " equivalent to ",  step_size*pred_horizon*1/50, "seconds")
# print('Dataset: ', dataset_name)

# translation = translation.cpu().detach().numpy()
# position_groundtruth = unnormalize_position(batch['position'], translation, stats['position'])
# position_prediction_history = []

# # # ==================================================== #
# # Run Environment:

# action_loaded = batch['action'].squeeze().cpu().detach().numpy()
# action = action_loaded[0]
# print("*** Running environment for actions ***")
# for i in range(NUM_RUNS):
#     print()
#     print("*** Sample from model ***")
#     x_0_predicted, _, _ = model.sample(batch=batch, mode='validation')
#     print("      ... Sampled ***")
#     print("*** Unnormalize sample ***")
#     x_0_predicted = x_0_predicted.squeeze().cpu().detach().numpy()
#     nAction_predicted = x_0_predicted[:, 2:5]
#     action_predicted = unnormalize_data(nAction_predicted, stats['action'])
#     print("      ... Unnormalized ***")

#     # -----  Initialize environment ----- #
#     env = CarRacing()
#     env.seed(ENV_SEED)
#     env.reset()
#     obs, _ , _, info = env.step(action) # Take a step to get the environment initialized (action is empty)

#     # Drive to the starting point and drive to obs_horizon
#     position_history = []
#     for i in range(start_idx + obs_horizon):
#         env.render(MODE)
#         augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
#             # Update the action every step_size steps
#         obs, _ , _, info = env.step(action_loaded[i])

#     # for i in range(pred_horizon * step_size):
#     #     env.render(MODE)
#     #     augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
#     #     if i % step_size == 0:
#     #         action = action_predicted[i//step_size]
#     #     obs, _ , _, info = env.step(action)
#     #     position_history.append(carPosition_wFrame.copy())
#     # position_history = np.array(position_history)

#     env.close()

# # Plotting
# print("*** Plotting ***")
# fig, ax = plt.subplots()
# plt.plot(position_history[:, 0], position_history[:, 1], color='green')
# plt.plot(position_groundtruth[:, 0], position_groundtruth[:, 1], color='red')
# plt.title("Position history")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend(['Predicted', 'Groundtruth'])
# plt.show()










# ---------------------------------------------------- #

# import matplotlib.pyplot as plt
# import yaml
# import sys
# import zarr
# import numpy as np

# # setting path
# sys.path.append('../diffusion_bare')

# from envs.car_racing import CarRacing
# from utils.data_utils import *
# from generateData.trajectory_control_utils import *
# from models.diffusion_ddpm import *

# # ###### GLOBAL VARIABLES ######
# SEED = 42
# MODE = 'state_pixels'
# ENV_SEED = 42
# EXPERIMENT_NAME = 'Consistency evaluations'


# # # paths
# dataset_path = './data/2023-07-20-1827_dataset_1_episodes_2_modes.zarr.zip'
# # dataset_path = "./evaluation/data/2023-07-21-1537/EvaluationDataset_left_dataset_1_episodes_1_modes.zarr.zip"
# # filepath = './tb_logs/version_629/STATS.pkl'
# # path_checkpoint = './tb_logs/version_629/checkpoints/epoch=39.ckpt'
# # path_hyperparams = './tb_logs/version_629/hparams.yaml'



# # path_hyperparams = './tb_logs/version_624/hparams.yaml'
# # path_checkpoint = './tb_logs/version_624/checkpoints/epoch=35.ckpt'
# # filepath = './tb_logs/version_624/STATS.pkl'
# # #dataset_name = '2023-07-15-1711_dataset_1_episodes_2_modes.zarr.zip'
# # dataset_path = './data/2023-07-20-1827_dataset_1_episodes_2_modes.zarr.zip'


# # dataset_path = "./evaluation/data/2023-07-21-1537/EvaluationDataset_left_dataset_1_episodes_1_modes.zarr.zip"
# filepath = './tb_logs/version_630/STATS.pkl'
# path_checkpoint = './tb_logs/version_630/checkpoints/epoch=41.ckpt'
# path_hyperparams = './tb_logs/version_630/hparams.yaml'

# # dataset_path = "./evaluation/data/2023-07-21-1537/EvaluationDataset_left_dataset_1_episodes_1_modes.zarr.zip"
# # filepath = './tb_logs/version_631/STATS.pkl'
# # path_checkpoint = './tb_logs/version_631/checkpoints/epoch=48.ckpt'
# # path_hyperparams = './tb_logs/version_631/hparams.yaml'


# # Settings
# # np.random.seed(SEED)


# # Define functions
# def fetch_hyperparams_from_yaml(file_path):
#     with open(file_path, 'r') as file:
#         hyperparams = yaml.safe_load(file)
#     return hyperparams

# def normalize_position(sample, stats):
#     sample_normalized = normalize_data(sample, stats['position'])
#     translation_vec = sample_normalized[0, :]
#     nsample_centered = sample_normalized - translation_vec
#     nsample = nsample_centered / 2.0
#     return nsample, translation_vec

# # ==================================================== #
# # ======================  MAIN  ====================== #
# # ==================================================== #

# print("*** Loading Data ...")
# dataset_root = zarr.open(dataset_path, 'r')
# data = {
#     'image': np.moveaxis(dataset_root['data']['img'][:], -1, 1),
#     'position': dataset_root['data']['position'][:],
#     'velocity': dataset_root['data']['velocity'][:],
#     'action': dataset_root['data']['action'][:],
# }
# model_params = fetch_hyperparams_from_yaml(path_hyperparams)
# obs_horizon = model_params['obs_horizon']
# pred_horizon = model_params['pred_horizon']
# inpaint_horizon = model_params['inpaint_horizon']
# print("     ... Data Loaded ***")

# print("*** Loading Stats ...")
# with open(filepath, 'rb') as f:
#     stats = pickle.load(f)[0]
# print("     ... Stats Loaded ***")

# print("*** Generate indices of possible trajectory samples ***")
# sequence_len = obs_horizon + pred_horizon
# indices = create_sample_indices(
#     episode_ends=[len(data['position'])],
#     sequence_length=sequence_len,
#     pad_before=0,
#     pad_after=0
# )
# print("*** Randomly chosen trajectory sample ...")
# buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx =  indices[520]  # indices[np.random.randint(0, len(indices))]
# sample = sample_sequence(
#     train_data= data,
#     sequence_length= sequence_len,
#     buffer_start_idx= buffer_start_idx,
#     buffer_end_idx= buffer_end_idx,
#     sample_start_idx= sample_start_idx,
#     sample_end_idx= sample_end_idx
# )
# start_t = buffer_start_idx
# end_t = buffer_end_idx
# print("     ... Trajectory sample obtained ***")

# print("-" * 10 )
# print(" -- Lenght of sequence: ", len(sample['position']))
# print(" -- Buffer start index: ", buffer_start_idx)
# print(" -- Buffer end index: ", buffer_end_idx)
# print("-" * 10 )

# print( "*** Normalize sample ...")
# nAction = normalize_data(sample['action'], stats=stats['action'])
# nVelocity = normalize_data(sample['velocity'], stats=stats['velocity'])
# nPosition, Translation = normalize_position(sample['position'], stats)

# nsample = {
#     'position': nPosition,
#     'velocity': nVelocity,
#     'action': nAction,
#     'image': sample['image'], # already normalized
# }
# print("     ... Sample normalized ***")


# print("*** Loading Model ...")
# # -----  Loading and initializing model ----- #
# model = Diffusion_DDPM.load_from_checkpoint(
#     path_checkpoint,
#     hparams_file=path_hyperparams,
# )
# model.eval()
# print("     ... Model Loaded ***")

# print("*** Create Batch ... ")
# batch = {
#     'position': torch.from_numpy(nsample['position']).unsqueeze(0).float(),
#     'velocity': torch.from_numpy(nsample['velocity']).unsqueeze(0).float(),
#     'action': torch.from_numpy(nsample['action']).unsqueeze(0).float(),
#     'image': torch.from_numpy(nsample['image']).unsqueeze(0).float(),
# }
# for key, value in batch.items():
#     print(key, value.shape)
# print("... Batch created ***")


# results_actions_pred = []
# results_actions_trajectories = []
# results_positions_pred = []

# # ================================================================================= #
# # ======================  Run N runs of the same experiment  ====================== #
# # ================================================================================= #

# NUM_RUNS = 50

# for i in range(NUM_RUNS):
#     print()
#     print("*** Sample from model ***")
#     x_0_predicted, _, _ = model.sample(batch=batch, mode='validation')
#     print("      ... Sampled ***")

#     print("*** Unnormalize sample ***")
#     x_0_predicted = x_0_predicted.squeeze().cpu().detach().numpy()
#     nAction_predicted = x_0_predicted[:, 2:5]
#     action_pred = unnormalize_data(nAction_predicted, stats['action']) # Includes inpainted points
#     nPosition_predicted = x_0_predicted[:, 0:2]
#     position_pred = unnormalize_data(2* nPosition_predicted + Translation, stats['position']) # Includes inpainted points
#     action_pred = action_pred[inpaint_horizon:]
#     position_pred = position_pred[:]
#     print("     ...Unnormalized ***")

#     # -----  Initialize environment ----- #
#     env = CarRacing()
#     env.seed(ENV_SEED)
#     env.reset()
#     action = np.array([0, 0, 0], dtype=np.float32)
#     obs, _ , _, info = env.step(action) # Take a step to get the environment initialized (action is empty)
#     # -----  Initialize history lists ----- #
#     img_hist, vel_hist ,act_hist, pos_hist, angle_hist = [], [], [], [], []


#     for i in range(start_t + obs_horizon):
#         env.render(MODE)
#         augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
#         obs, _ , _, info = env.step(data['action'][i])

#     for i in range(pred_horizon):
#         env.render(MODE)
#         augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
#         obs, _ , _, info = env.step(action_pred[i])
#         append_to_histories(obs, carVelocity_wFrame, carPosition_wFrame, action, car_heading_angle, img_hist, vel_hist, pos_hist, act_hist, angle_hist)
#     pos_hist = np.array(pos_hist)
#     actions_saved_traj = pos_hist.copy()
#     env.close()
#     results_actions_trajectories.append(actions_saved_traj.copy())
#     results_actions_pred.append(action_pred.copy())
#     results_positions_pred.append(position_pred[inpaint_horizon:].copy())


# # ================================================================================= #
# # ======================  Plotting the results  =================================== #
# # ================================================================================= #


# # Plot predicted trajectories
# plt.figure(figsize=(10, 10))
# plt.title("Predicted trajectories")
# plt.xlabel("x")
# plt.ylabel("y")

# for i in range(NUM_RUNS):
#     plt.plot(results_positions_pred[i][:, 0], results_positions_pred[i][:, 1], label=f'Run {i}')
#     plt.scatter(results_positions_pred[i][:, 0], results_positions_pred[i][:, 1], s=10)
# plt.plot(data['position'][start_t:end_t, 0], data['position'][start_t:end_t, 1], label='Ground truth trajectory')


# plt.legend()
# plt.show()

# # Pointwise distance between predicted and ground truth trajectories
# distances = []
# for i in range(NUM_RUNS):
#     distances.append(np.linalg.norm(results_positions_pred[i] - results_actions_trajectories[i], axis=1))
# distances = np.array(distances)
# print(distances.shape)

# distances_mean = np.mean(distances, axis=0)
# print(distances_mean.shape)


# # Plot pointwise distances, meaning error at each timestep
# plt.figure(figsize=(10, 10))
# plt.title("Pointwise distance between predicted and ground truth trajectories, 50 Runs")
# plt.xlabel("timestep")
# plt.ylabel("distance")
# plt.plot(distances_mean, label='Mean', linewidth=3)
# # for i in range(NUM_RUNS):
# #     plt.plot(distances[i], label=f'Run {i}')
# plt.legend()
# plt.show()



