import sys
import pickle
import yaml
import torch
import numpy as np
import zarr
import matplotlib.pyplot as plt

# setting path
sys.path.append('../diffusion_bare')

from models.diffusion_ddpm import *
from models.diffusion_ddim import *
from utils.load_data import *

# ###### GLOBAL VARIABLES ######
NUM_RUNS = 2
SEED = 42
MODE = 'state_pixels'
ENV_SEED = 42
EXPERIMENT_NAME = 'Consistency evaluations' 
MODEL_NAME = 'DDPM'

def load_model(model_name, checkpoint_path, hparams_path):
    if model_name == 'DDPM':
        model = Diffusion_DDPM.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path)
    elif model_name == 'DDIM' or model_name == 'DDIPM':
        model = Diffusion_DDIM.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path)
    else:
        raise ValueError("Model name must be either 'DDPM', 'DDIM', or 'DDIPM'")
    model.eval()
    return model

# # paths
dataset_path = './data/'
dataset_name = '2023-07-17-1052_dataset_1_episodes_2_modes.zarr.zip'
filepath = './tb_logs/version_664/STATS.pkl'
path_checkpoint = './tb_logs/version_664/checkpoints/epoch=12.ckpt'
path_hyperparams = './tb_logs/version_664/hparams.yaml'

with open(filepath, 'rb') as f:
    stats = pickle.load(f)
stats = stats[0]

#Model
model = load_model(MODEL_NAME,  path_checkpoint, path_hyperparams)
model_params = fetch_hyperparams_from_yaml(path_hyperparams)
obs_horizon = model_params['obs_horizon']
pred_horizon = model_params['pred_horizon']
# Dataloader
dataset = CarRacingDataModule(1 , dataset_path, obs_horizon, pred_horizon, seed=2, stats=stats)
dataset.setup(name=dataset_name)
test_dataloaders = dataset.val_dataloader()
batch, translation = next(iter(test_dataloaders))
step_size = model_params['step_size']
# ==================================================== #
# Print hyperparameter information
print('Hyperparameters:')
print('Model: ', MODEL_NAME)
print('Step size: ', step_size , 'equivalent to ', step_size*1/50, 'seconds')
print('Obs horizon: ', obs_horizon , " equivalent to ", step_size*obs_horizon*1/50, "seconds")
print('Pred horizon: ', pred_horizon , " equivalent to ",  step_size*pred_horizon*1/50, "seconds")
print('Dataset: ', dataset_name)


# ==================================================== #
# Plot dataset sample
# fig, ax = plt.subplots()
# plt.plot(batch['position'][0, :, 0], batch['position'][0, :, 1], 'o')
# plt.plot(batch['position'][0, :obs_horizon, 0], batch['position'][0, :obs_horizon, 1], 'o', color='red')
# plt.title('Dataset sample')
# plt.show()

# ==================================================== #
# Environment execution and plotting
translation = translation.cpu().detach().numpy()
# Unormalize position in batch
# position_groundtruth = unnormalize_position(batch['position'], translation, stats['position'])
position_groundtruth = batch['position']
   
position_prediction_history = [] 
for i in range(NUM_RUNS):
    print()
    print("*** Sample from model ***")
    x_0_predicted, _, _ = model.sample(batch=batch, mode='validation')
    print("      ... Sampled ***")

    # print("*** Unnormalize sample ***")
    x_0_predicted = x_0_predicted.squeeze().cpu().detach().numpy()

    # nAction_predicted = x_0_predicted[:, 2:5]
    nPosition_predicted = x_0_predicted[:, 0:2]
    # position_pred = unnormalize_position(nPosition_predicted , translation, stats['position']) 
    position_pred = nPosition_predicted
    position_prediction_history.append(position_pred.copy())
    # print("     ...Unnormalized ***")
print("*** Plotting ***")


fig, ax = plt.subplots()
plt.plot(position_groundtruth[0, :, 0], position_groundtruth[0, :, 1])
plt.plot(position_groundtruth[0, :obs_horizon, 0],position_groundtruth[0, :obs_horizon, 1], 'o', color='red')

for i in range(NUM_RUNS):
    plt.plot(position_prediction_history[i][:, 0], position_prediction_history[i][:, 1], color='green')
    plt.scatter(position_prediction_history[i][:, 0], position_prediction_history[i][:, 1], color='green' , s=10)
plt.title('Sample from model')
ax.set_aspect('equal')
# Get the current limits
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# Find the range of the data
data_range = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
# Set the limits to the range
ax.set_xlim(data_range)
ax.set_ylim(data_range)

plt.show()
















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



