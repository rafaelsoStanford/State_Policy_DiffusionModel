import matplotlib.pyplot as plt
import yaml
import sys
import zarr
import numpy as np

# setting path
sys.path.append('../diffusion_bare')

from envs.car_racing import CarRacing
from utils.data_utils import *
from generateData.trajectory_control_utils import *
from models.diffusion_ddpm import *

# ###### GLOBAL VARIABLES ######
NUM_RUNS = 2
MODE = 'state_pixels'
SEED = 42
ENV_SEED = 42
EXPERIMENT_NAME = 'Precision evaluations' 
NUM_RUNS = 2
STEP_SIZE = 5
DATASET_PATH = './data/2023-07-20-1827_dataset_1_episodes_2_modes.zarr.zip'
STATS_FILEPATH = './tb_logs/version_654/STATS.pkl'
CHECKPOINT_PATH = './tb_logs/version_654/checkpoints/epoch=12.ckpt'
HYPERPARAMS_PATH = './tb_logs/version_654/hparams.yaml'
step_size = 10

# Fix the seed for reproducibility
np.random.seed(SEED)


def normalize_position(sample, stats):
    sample_normalized = normalize_data(sample, stats['position'])
    translation_vec = sample_normalized[0, :]
    nsample_centered = sample_normalized - translation_vec
    nsample = nsample_centered / 2.0
    return nsample, translation_vec

# Load the data
dataset_root = zarr.open(DATASET_PATH, 'r')
data = {
    'image': np.moveaxis(dataset_root['data']['img'][:], -1, 1),
    'position': dataset_root['data']['position'][:],
    'velocity': dataset_root['data']['velocity'][:],
    'action': dataset_root['data']['action'][:],
}

# Load the model's hyperparameters
model_params = fetch_hyperparams_from_yaml(HYPERPARAMS_PATH)
obs_horizon = model_params['obs_horizon']
pred_horizon = model_params['pred_horizon']
inpaint_horizon = model_params['inpaint_horizon']

# Load statistics
with open(STATS_FILEPATH, 'rb') as f:
    stats = pickle.load(f)[0]

# Generate indices of possible trajectory samples
sequence_len = obs_horizon + pred_horizon
indices = create_sample_indices_sparse(
    ends=[len(data['position'])],
    sequence_length=sequence_len,
    step_size=STEP_SIZE
)

# Randomly choose a trajectory sample
start_t, end_t, sample_start_idx, sample_end_idx = indices[950] 
sample = sample_sequence_sparse(
    data= data,
    step_size=STEP_SIZE,
    sample_start_idx=start_t,
    sample_end_idx=end_t,
)

# Normalize the sample
nAction = normalize_data(sample['action'], stats=stats['action'])
nVelocity = normalize_data(sample['velocity'], stats=stats['velocity'])
nPosition, Translation = normalize_position(sample['position'], stats)

nsample = {
    'position': nPosition,
    'velocity': nVelocity,
    'action': nAction,
    'image': sample['image'], # already normalized
}

# Load the model
model = Diffusion_DDPM.load_from_checkpoint(CHECKPOINT_PATH, hparams_file=HYPERPARAMS_PATH)
model.eval()

# Create a batch
batch = {
    'position': torch.from_numpy(nsample['position']).unsqueeze(0).float(),
    'velocity': torch.from_numpy(nsample['velocity']).unsqueeze(0).float(),
    'action': torch.from_numpy(nsample['action']).unsqueeze(0).float(),
    'image': torch.from_numpy(nsample['image']).unsqueeze(0).float(),
}

results_actions_pred = []
results_actions_trajectories = []
results_positions_pred = []

# Run multiple runs of the same experiment
for i in range(NUM_RUNS):
    x_0_predicted, _, _ = model.sample(batch=batch, mode='validation')
    
    # Unnormalize the sample
    x_0_predicted = x_0_predicted.squeeze().cpu().detach().numpy()
    nAction_predicted = x_0_predicted[:, 2:5]
    action_pred = unnormalize_data(nAction_predicted, stats['action']) # Includes inpainted points
    nPosition_predicted = x_0_predicted[:, 0:2]
    position_pred = unnormalize_data(2* nPosition_predicted + Translation, stats['position']) # Includes inpainted points
    action_pred = action_pred[inpaint_horizon:]
    position_pred = position_pred[:]

    # Initialize the environment
    env = CarRacing()
    env.seed(ENV_SEED)
    env.reset()
    action = np.array([0, 0, 0], dtype=np.float32)
    obs, _ , _, info = env.step(action) # Take a step to get the environment initialized (action is empty)

    # Initialize history lists
    img_hist, vel_hist ,act_hist, pos_hist, angle_hist = [], [], [], [], []

    for i in range(start_t, end_t):
        env.render(MODE)
        augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
        
        # execute an action every step_size times, otherwise keep the action unchanged, this corresponds to zero-order hold
        if i % step_size == 0:
            idx = (i-start_t) // step_size
            if idx < len(action_pred):
                action = action_pred[idx]
        obs, _ , _, info = env.step(action) 
        append_to_histories(obs, carVelocity_wFrame, carPosition_wFrame, action, car_heading_angle, img_hist, vel_hist, pos_hist, act_hist, angle_hist)

    pos_hist = np.array(pos_hist)
    actions_saved_traj = pos_hist.copy()
    env.close()
    results_actions_trajectories.append(actions_saved_traj.copy())
    results_actions_pred.append(action_pred.copy())
    results_positions_pred.append(position_pred.copy())

# Extract groundtruth trajectory
groundtruth_trajectory = data['position'][start_t:end_t, :]

# Plotting
plt.figure(figsize=(10, 10))
plt.title("Trajectories Generated by Executing Actions")
plt.scatter(groundtruth_trajectory[:,0], groundtruth_trajectory[:,1], color='b', label='Groundtruth')
for i, traj in enumerate(results_actions_trajectories):
    plt.plot(traj[:,0], traj[:,1], '--', label=f'Run {i+1}')
plt.legend()
plt.show()

plt.figure(figsize=(10, 10))
plt.title("Sampled Position Points")
plt.scatter(groundtruth_trajectory[:,0], groundtruth_trajectory[:,1], color='b', label='Groundtruth')
for i, pos in enumerate(results_positions_pred):
    plt.scatter(pos[:,0], pos[:,1], marker='x', label=f'Run {i+1}')
plt.legend()
plt.show()



# # ###### GLOBAL VARIABLES ######
# SEED = 42
# MODE = 'state_pixels'
# ENV_SEED = 42
# EXPERIMENT_NAME = 'Precision evaluations' 
# NUM_RUNS = 2
# step_size = 5

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
# # filepath = './tb_logs/version_630/STATS.pkl'
# # path_checkpoint = './tb_logs/version_630/checkpoints/epoch=41.ckpt'
# # path_hyperparams = './tb_logs/version_630/hparams.yaml'

# # dataset_path = "./evaluation/data/2023-07-21-1537/EvaluationDataset_left_dataset_1_episodes_1_modes.zarr.zip"
# # filepath = './tb_logs/version_631/STATS.pkl'
# # path_checkpoint = './tb_logs/version_631/checkpoints/epoch=48.ckpt'
# # path_hyperparams = './tb_logs/version_631/hparams.yaml'


# filepath = './tb_logs/version_659/STATS.pkl'
# path_checkpoint = './tb_logs/version_659/checkpoints/epoch=12.ckpt'
# path_hyperparams = './tb_logs/version_659/hparams.yaml'


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
# indices = create_sample_indices_sparse(
#     ends=[len(data['position'])],
#     sequence_length=sequence_len,
#     step_size=5
#     # pad_before=0,
#     # pad_after=0
# )
# print("*** Randomly chosen trajectory sample ...")
# buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx =  indices[950]  # indices[np.random.randint(0, len(indices))]
# sample = sample_sequence_sparse(
#     data= data,
#     step_size=5,
#     sample_start_idx= buffer_start_idx,
#     sample_end_idx= buffer_end_idx,
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
#     results_positions_pred.append(position_pred.copy())


# ================================================================================= #
# ======================  Plotting the results  =================================== #
# ================================================================================= #

# SAVING = False

# plt.figure(figsize=(20, 20))

# for i in range(NUM_RUNS):
#     if i == 0:
#         plt.plot(results_actions_trajectories[i][:, 0], results_actions_trajectories[i][:, 1], label='Predicted trajectory', c = 'r', linewidth=1)
#         plt.scatter(results_actions_trajectories[i][:, 0], results_actions_trajectories[i][:, 1], c = 'r', s=10, marker='x')
#     else:
#         plt.plot(results_actions_trajectories[i][:, 0], results_actions_trajectories[i][:, 1], c = 'r', linewidth=1)
#         plt.scatter(results_actions_trajectories[i][:, 0], results_actions_trajectories[i][:, 1], c = 'r', s=10, marker='x')

# plt.title('Predicted trajectories using actions from diffusion model')
# text_str = f"Horizon parameters:    Observation Horizon {obs_horizon}\n  Prediction Horizon {pred_horizon}\n    Inpaint Horizon {inpaint_horizon}"
# plt.text(0.35, 0.05, text_str, transform=plt.gca().transAxes, ha='right', va='bottom', fontsize=12)
# plt.plot(data['position'][start_t:end_t:step_size, 0], data['position'][start_t:end_t:step_size, 1], label='Actual trajectory', c = 'b')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


# if SAVING:
#     folder_name = folder_name = "./evaluation/results/eval_precision"  # You can change this to the desired folder name
#     # Get the horizon information
#     horizon_info = f"obs_{obs_horizon}_pred_{pred_horizon}_inpaint_{inpaint_horizon}"
#     # Specify the file name for the plot (with extension)
#     plot_file_name = f"predicted_actions_trajectories_{horizon_info}.png"  # You can change the file name if needed
#     # Save the plot to the specified folder with the given file name
#     plt.savefig(os.path.join(folder_name, plot_file_name))
# plt.show()

# plt.figure(figsize=(20, 20))
# for i in range(NUM_RUNS):
#     if i == 0:
#         plt.plot(results_positions_pred[i][inpaint_horizon-1:, 0], results_positions_pred[i][inpaint_horizon-1:, 1], label='Predicted trajectory', c = 'orange', linewidth=1)
#         plt.scatter(results_positions_pred[i][inpaint_horizon:, 0], results_positions_pred[i][inpaint_horizon:, 1], c = 'orange', s=10, marker='x')
#         plt.scatter(results_positions_pred[i][:inpaint_horizon, 0], results_positions_pred[i][:inpaint_horizon, 1], label='Inpainted trajectory points', c = 'k', s=10, marker='x')
#     else:
#         plt.plot(results_positions_pred[i][inpaint_horizon-1:, 0], results_positions_pred[i][inpaint_horizon-1:, 1], c = 'orange', linewidth=1)
#         plt.scatter(results_positions_pred[i][inpaint_horizon:, 0], results_positions_pred[i][inpaint_horizon:, 1], c = 'orange', s=10, marker='x')
        
# plt.plot(data['position'][start_t:end_t:step_size, 0], data['position'][start_t:end_t:step_size, 1], label='Actual trajectory', c = 'b')
# plt.title('Predicted trajectories using actions from diffusion model')
# text_str = f"Horizon parameters:    Observation Horizon {obs_horizon}\n  Prediction Horizon {pred_horizon}\n    Inpaint Horizon {inpaint_horizon}"
# plt.text(0.5, 0.05, text_str, transform=plt.gca().transAxes, ha='right', va='bottom', fontsize=12)

# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')

# if SAVING:
#     # Specify the file name for the plot (with extension)
#     plot_file_name = f"predicted_trajectories_{horizon_info}.png"  # You can change the file name if needed
#     # Save the plot to the specified folder with the given file name
#     plt.savefig(os.path.join(folder_name, plot_file_name))
# plt.show()


# # Calculate squared errors for each point in the trajectory
# error_sq_array = []
# for trajectory in results_actions_trajectories:
#     distance = trajectory - data['position'][start_t + obs_horizon:end_t:step_size]
#     norm = np.linalg.norm(distance, axis=1) ** 2
#     error_sq_array.append(norm.copy())
# error_sq_array = np.array(error_sq_array)

# # Calculate the mean standard deviation along the trajectory
# std_along_traj = np.sqrt(1 / (NUM_RUNS - 1) * np.sum(error_sq_array, axis=0))

# # Calculate standard deviation for each time step in both x and y axes
# std_x_list = []
# std_y_list = []
# for trajectory in results_actions_trajectories:
#     distance = trajectory - data['position'][start_t + obs_horizon:end_t:step_size]
#     error_x = distance[:, 0] ** 2
#     std_x_list.append(error_x.copy())
#     error_y = distance[:, 1] ** 2
#     std_y_list.append(error_y.copy())

# std_x_along_traj = np.sqrt(1 / (NUM_RUNS - 1) * np.sum(np.array(std_x_list), axis=0))
# std_y_along_traj = np.sqrt(1 / (NUM_RUNS - 1) * np.sum(np.array(std_y_list), axis=0))

# # Plot mean standard deviation along the trajectory
# plt.figure(figsize=(20, 20))
# plt.plot(std_along_traj, label='Mean Standard Deviation')
# plt.title('Mean Standard Deviation Along the Trajectory')
# text_str = f"Horizon parameters:    Observation Horizon {obs_horizon}\nPrediction Horizon {pred_horizon}\nInpaint Horizon {inpaint_horizon}"
# plt.text(0.5, 0.05, text_str, transform=plt.gca().transAxes, ha='right', va='bottom', fontsize=12)
# plt.xlabel('Time Step')
# plt.ylabel('Standard Deviation')
# plt.legend()


# if SAVING:
#     # Specify the file name for the plot (with extension)
#     plot_file_name = f"mean_std_deviation{horizon_info}.png"  # You can change the file name if needed
#     # Save the plot to the specified folder with the given file name
#     plt.savefig(os.path.join(folder_name, plot_file_name))
# plt.show()


# # Plot predicted trajectory with error bars
# plt.figure(figsize=(20, 20))
# plt.errorbar(data['position'][start_t + obs_horizon:end_t:step_size, 0], data['position'][start_t + obs_horizon:end_t:step_size, 1],
#              yerr=std_y_along_traj, xerr=std_x_along_traj, fmt='o', color='r', alpha=0.7, label='Predicted Trajectory')
# plt.plot(data['position'][start_t:end_t:step_size, 0], data['position'][start_t:end_t:step_size, 1], label='Actual Trajectory', c='b')
# plt.title('Predicted Trajectory with Error Bars')
# text_str = f"Horizon parameters:    Observation Horizon {obs_horizon}\nPrediction Horizon {pred_horizon}\nInpaint Horizon {inpaint_horizon}"
# plt.text(0.5, 0.05, text_str, transform=plt.gca().transAxes, ha='right', va='bottom', fontsize=12)
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')

# if SAVING:
#     # Specify the file name for the plot (with extension)
#     plot_file_name = f"Error_Bars2d{horizon_info}.png"  # You can change the file name if needed
#     # Save the plot to the specified folder with the given file name
#     plt.savefig(os.path.join(folder_name, plot_file_name))
# plt.show()


