"""

Establish a robustness measure for the model. Adding noise to the observations and see how the model performs.


"""



import matplotlib.pyplot as plt
import yaml
import sys
import zarr
import numpy as np


# setting path
sys.path.append('../diffusion_bare')

from envs.car_racing import CarRacing
from utils.load_data import *
from generateData.trajectory_control_utils import *
from models.diffusion_ddpm import *

# ###### GLOBAL VARIABLES ######
SEED = 42
MODE = 'state_pixels'
ENV_SEED = 42
EXPERIMENT_NAME = 'Precision evaluations' 

# # paths
dataset_path = './data/2023-07-20-1827_dataset_1_episodes_2_modes.zarr.zip'
# dataset_path = "./evaluation/data/2023-07-21-1537/EvaluationDataset_left_dataset_1_episodes_1_modes.zarr.zip"
# filepath = './tb_logs/version_629/STATS.pkl'
# path_checkpoint = './tb_logs/version_629/checkpoints/epoch=39.ckpt'
# path_hyperparams = './tb_logs/version_629/hparams.yaml'



# path_hyperparams = './tb_logs/version_624/hparams.yaml'
# path_checkpoint = './tb_logs/version_624/checkpoints/epoch=35.ckpt'
# filepath = './tb_logs/version_624/STATS.pkl'
# #dataset_name = '2023-07-15-1711_dataset_1_episodes_2_modes.zarr.zip'
# dataset_path = './data/2023-07-20-1827_dataset_1_episodes_2_modes.zarr.zip'


# dataset_path = "./evaluation/data/2023-07-21-1537/EvaluationDataset_left_dataset_1_episodes_1_modes.zarr.zip"
filepath = './tb_logs/version_630/STATS.pkl'
path_checkpoint = './tb_logs/version_630/checkpoints/epoch=41.ckpt'
path_hyperparams = './tb_logs/version_630/hparams.yaml'

# dataset_path = "./evaluation/data/2023-07-21-1537/EvaluationDataset_left_dataset_1_episodes_1_modes.zarr.zip"
# filepath = './tb_logs/version_631/STATS.pkl'
# path_checkpoint = './tb_logs/version_631/checkpoints/epoch=48.ckpt'
# path_hyperparams = './tb_logs/version_631/hparams.yaml'



# Define functions
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

# ==================================================== #
# ======================  MAIN  ====================== #
# ==================================================== #

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
buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx =  indices[950]  # indices[np.random.randint(0, len(indices))]
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


print("*** Loading Model ...")
# -----  Loading and initializing model ----- #
model = Diffusion_DDPM.load_from_checkpoint(
    path_checkpoint,
    hparams_file=path_hyperparams,
)
model.eval()
print("     ... Model Loaded ***")

print("*** Create Batch ... ")
batch = {
    'position': torch.from_numpy(nsample['position']).unsqueeze(0).float(),
    'velocity': torch.from_numpy(nsample['velocity']).unsqueeze(0).float(),
    'action': torch.from_numpy(nsample['action']).unsqueeze(0).float(),
    'image': torch.from_numpy(nsample['image']).unsqueeze(0).float(),
}
for key, value in batch.items():
    print(key, value.shape)
print("... Batch created ***")


results_actions_pred = []
results_actions_trajectories = []
results_positions_pred = []

# ================================================================================= #
# ======================  Run N runs of the same experiment  ====================== #
# ================================================================================= #

NUM_RUNS = 10
factor = 0.01

# -----  Buffers holding the different trajectories ----- #

trajectory_buffer_position = np.zeros((NUM_RUNS, pred_horizon, 2))
trajectory_buffer_actions = np.zeros((NUM_RUNS, pred_horizon, 3))

for run in range(NUM_RUNS):
    scaling_factor = run * factor

    print(f"*** Run {run} ***")

    print(" Adding noise to the observations ...")
    noise_image = torch.rand_like(batch['image']) * scaling_factor
    batch['image'] = batch['image'] + noise_image

    noise_position = torch.rand_like(batch['position']) * scaling_factor
    batch['position'] = batch['position'] + noise_position

    noise_velocity = torch.rand_like(batch['velocity']) * scaling_factor
    batch['velocity'] = batch['velocity'] + noise_velocity

    noise_action = torch.rand_like(batch['action']) * scaling_factor
    batch['action'] = batch['action'] + noise_action
    print("     ... Noise added ***")


    print()
    print("*** Sample from model ***")
    x_0_predicted, _, _ = model.sample(batch=batch, mode='validation')
    print("      ... Sampled ***")

    print("*** Unnormalize sample ***")
    x_0_predicted = x_0_predicted.squeeze().cpu().detach().numpy()
    nAction_predicted = x_0_predicted[:, 2:5]
    action_pred = unnormalize_data(nAction_predicted, stats['action']) # Includes inpainted points
    nPosition_predicted = x_0_predicted[:, 0:2]
    position_pred = unnormalize_data(2* nPosition_predicted + Translation, stats['position']) # Includes inpainted points
    action_pred = action_pred[inpaint_horizon:]
    position_pred = position_pred[:]
    print("     ...Unnormalized ***")

    # -----  Initialize environment ----- #
    env = CarRacing()
    env.seed(ENV_SEED)
    env.reset()
    action = np.array([0, 0, 0], dtype=np.float32)
    obs, _ , _, info = env.step(action) # Take a step to get the environment initialized (action is empty)
    # -----  Initialize history lists ----- #
    img_hist, vel_hist ,act_hist, pos_hist, angle_hist = [], [], [], [], []

    
    for i in range(start_t + obs_horizon):
        env.render(MODE)
        augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
        obs, _ , _, info = env.step(data['action'][i]) 

    for i in range(pred_horizon):
        env.render(MODE)
        augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame = extract_variables(info)
        obs, _ , _, info = env.step(action_pred[i]) 
        append_to_histories(obs, carVelocity_wFrame, carPosition_wFrame, action, car_heading_angle, img_hist, vel_hist, pos_hist, act_hist, angle_hist)
    env.close()
    # -----  Save trajectories ----- #
    pos_hist = np.array(pos_hist)
    trajectory_buffer_position[run, :, :] = pos_hist.copy()
    trajectory_buffer_actions[run, :, :] = action_pred.copy()
    

# ================================================================================= #
# ======================  Plotting the results  =================================== #
# ================================================================================= #

# -----  Plotting the trajectories ----- #
plt.figure(figsize=(10, 10))
plt.title("Trajectories")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(data['position'][start_t:end_t, 0], data['position'][start_t:end_t, 1], label='Actual trajectory', c='b')

for plot in range(NUM_RUNS):
    alpha_val = 1 - (plot / NUM_RUNS * 0.8)

    if plot == 0:
        plt.plot(
            trajectory_buffer_position[plot, :, 0],
            trajectory_buffer_position[plot, :, 1],
            label='Noiseless prediction',
            linewidth=3,
            c='r'
        )
    else:
        plt.plot(
            trajectory_buffer_position[plot, :, 0],
            trajectory_buffer_position[plot, :, 1],
            label=f'alpha = {plot * factor}',
            alpha=alpha_val
        )

plt.legend()
plt.show()

# -----  Plotting the actions ----- #
fig, axs = plt.subplots(1, 3)
fig.suptitle('Actionspace: Prediction with noise N(0,1) * alpha \n to normalized observations')

axs[0].plot(data['action'][start_t+obs_horizon:end_t, 0], label='Groundtruth actions' ,linewidth=3)
legend2 = axs[0].legend(fontsize='small', loc='upper left')
axs[1].plot(data['action'][start_t+obs_horizon:end_t, 1], label='Groundtruth actions',linewidth=3)
legend4 = axs[1].legend(fontsize='small', loc='upper left')
axs[2].plot(data['action'][start_t+obs_horizon:end_t, 2], label='Groundtruth actions',linewidth=3)
legend6 = axs[2].legend(fontsize='small', loc='upper left')

for plot in range(NUM_RUNS):

    if plot == 0:
        axs[0].plot(trajectory_buffer_actions[plot, :, 0], label='Noiseless prediction', linewidth=3, c='r')
        legend2 = axs[0].legend(fontsize='small', loc='upper left')
        axs[1].plot(trajectory_buffer_actions[plot, :, 1], label='Noiseless prediction', linewidth=3, c='r')
        legend4 = axs[1].legend(fontsize='small', loc='upper left')
        axs[2].plot(trajectory_buffer_actions[plot, :, 2], label='Noiseless prediction', linewidth=3 , c='r')
        legend6 = axs[2].legend(fontsize='small', loc='upper left')

    else:
        alpha_val = 1 - (plot/NUM_RUNS*0.8)
        axs[0].plot(trajectory_buffer_actions[plot, :, 0], label='alpha = ' + str(plot* factor), alpha=alpha_val)
        legend2 = axs[0].legend(fontsize='small', loc='upper left')
        axs[1].plot(trajectory_buffer_actions[plot, :, 1], label='alpha = ' + str(plot* factor), alpha=alpha_val)
        legend4 = axs[1].legend(fontsize='small', loc='upper left')
        axs[2].plot(trajectory_buffer_actions[plot, :, 2], label='alpha = ' + str(plot* factor), alpha=alpha_val)
        legend6 = axs[2].legend(fontsize='small', loc='upper left')

axs[0].set_title('Steering')
axs[1].set_title('Acceleration')
axs[2].set_title('Brake')

axs[0].set_xlabel('Time')
axs[1].set_xlabel('Time')
axs[2].set_xlabel('Time')

axs[0].set_ylabel('Steering input value')
axs[1].set_ylabel('Acceleration input value')
axs[2].set_ylabel('Brake input value')

plt.show()

# Calculate the MSE between the groundtruth and the predicted trajectory for all noise levels
mse_position = np.zeros(NUM_RUNS)
mse_actions = np.zeros(NUM_RUNS)
for run in range(NUM_RUNS):
    mse_position[run] = np.mean(np.square(trajectory_buffer_position[run, :, :] - data['position'][start_t+obs_horizon:start_t+obs_horizon+pred_horizon, :]))
    mse_actions[run] = np.mean(np.square(trajectory_buffer_actions[run, :, :] - data['action'][start_t+obs_horizon:start_t+obs_horizon+pred_horizon, :]))
    



