import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Setting path
sys.path.append('../diffusion_bare')

from envs.car_racing import CarRacing
from models.diffusion_ddpm import *
from models.diffusion_ddim import *
from utils.load_data import *
from utils.data_utils import *
from generateData.trajectory_control_utils import *

def load_model(model_name, checkpoint_path, hparams_path):
    """Load the model from checkpoint and hparams path based on the model name"""
    if model_name == 'DDPM':
        model = Diffusion_DDPM.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path)
    elif model_name == 'DDIM' or model_name == 'DDIPM':
        model = Diffusion_DDIM.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path)
    else:
        raise ValueError("Model name must be either 'DDPM', 'DDIM', or 'DDIPM'")
    model.eval()
    return model




dataset_path = './evaluation/data/'
dataset_name = 'Evaluation_dataset_1_episodes_1_modes.zarr.zip'
filepath = './tb_logs/version_671/STATS.pkl'
path_checkpoint = './tb_logs/version_671/checkpoints/epoch=10.ckpt'
path_hyperparams = './tb_logs/version_671/hparams.yaml'

dataset_root = zarr.open(dataset_path + dataset_name, 'r')
data = {'image': np.moveaxis(dataset_root['data']['img'][:], -1, 1), 'position': dataset_root['data']['position'][:], 'velocity': dataset_root['data']['velocity'][:], 'action': dataset_root['data']['action'][:]}

model_params = fetch_hyperparams_from_yaml(path_hyperparams)
obs_horizon, pred_horizon, inpaint_horizon, step_size = model_params['obs_horizon'], model_params['pred_horizon'], model_params['inpaint_horizon'], model_params['step_size']

with open(filepath, 'rb') as f:
    stats = pickle.load(f)[0]

model = Diffusion_DDPM.load_from_checkpoint(path_checkpoint, hparams_file=path_hyperparams)
model.eval()

sequence_len = obs_horizon + pred_horizon
indices = create_sample_indices_sparse(ends=[data['position'].shape[0]], sequence_length=sequence_len, step_size=step_size)

start_idx, end_idx, _, _ = indices[70]
sample = sample_sequence_sparse(data=data, step_size=step_size, sample_start_idx=start_idx, sample_end_idx=end_idx)

nAction = normalize_data(sample['action'], stats=stats['action'])
nVelocity = normalize_data(sample['velocity'], stats=stats['velocity'])
nPosition, Translation = normalize_position(sample['position'], stats['position'])

nsample = {'position': nPosition, 'velocity': nVelocity, 'action': nAction, 'image': sample['image']}

batch = {'position': torch.from_numpy(nsample['position']).unsqueeze(0).float(), 
         'velocity': torch.from_numpy(nsample['velocity']).unsqueeze(0).float(), 
         'action': torch.from_numpy(nsample['action']).unsqueeze(0).float(), 
         'image': torch.from_numpy(nsample['image']).unsqueeze(0).float()
        }

position_prediction_history = [] 
print("*** Sampling for position estimations***")
NUM_RUNS = 10
MODEL_NAME = 'DDPM'

for i in range(NUM_RUNS):
    predicted_sample, _, _ = model.sample(batch=batch, mode='validation')
    predicted_sample = predicted_sample.squeeze().cpu().detach().numpy()
    nPosition_predicted = predicted_sample[:, 0:2]
    position_pred = unnormalize_position(nPosition_predicted , Translation , stats['position']) 
    position_prediction_history.append(position_pred.copy())
position_groundtruth = unnormalize_position(nsample['position'], Translation, stats['position'])

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================


fig, ax = plt.subplots()
for i in range(len(position_prediction_history)):
    plt.plot(position_prediction_history[i][:, 0], position_prediction_history[i][:, 1], color='green')
    plt.scatter(position_prediction_history[i][:, 0], position_prediction_history[i][:, 1], color='green' , s=10)
plt.plot(position_groundtruth[:, 0], position_groundtruth[:, 1])
plt.plot(position_groundtruth[:obs_horizon, 0], position_groundtruth[ :obs_horizon, 1], 'o', color='red')
plt.title('Diffusion position predicitons, with ' + MODEL_NAME + ' model, \n Horizons: ' + str(obs_horizon) + ' obs, ' + str(pred_horizon) + ' pred, Step size: ' + str(step_size*1/50) + ' seconds')
# Equal aspect ratio
# ax.set_aspect('equal', 'box')
plt.axis('scaled')

plt.show()


# Initialize a figure for the error plot
fig_error, ax_error = plt.subplots()

# Initialize a list to store all errors for averaging
all_errors = []

flag = False

# Loop over each predicted trajectory
for i in range(len(position_prediction_history)):
    # Calculate the Euclidean distance between the predicted points and the ground truth
    error = np.sqrt(np.sum((position_prediction_history[i][inpaint_horizon:] - position_groundtruth[ inpaint_horizon:, :2])**2, axis=1))
    # Store the error
    all_errors.append(error)
    # Plot the error over time in red and slightly transparent, only label the first one
    if flag is False:
        ax_error.plot(np.arange(len(error))*step_size / 50.0, error, color='r', alpha=0.5, label=f'Run 1 to {len(position_prediction_history)}')
        flag = True
    else:
        ax_error.plot(np.arange(len(error))*step_size / 50.0, error, color='r', alpha=0.5)

# Convert all errors to a numpy array for ease of calculation
all_errors = np.array(all_errors)

# Calculate the average error
average_error = np.mean(all_errors, axis=0)

# Calculate the standard deviation of error
std_error = np.std(all_errors, axis=0)

# Plot the average error over time in blue, bold
ax_error.plot(np.arange(len(average_error))*step_size / 50.0, average_error, color='b', linewidth=2, label='Average Error')

# Fill the area between the average plus standard deviation and the average minus standard deviation
ax_error.fill_between(np.arange(len(average_error))*step_size / 50.0, average_error - std_error, average_error + std_error, color='b', alpha=0.2)

# Formatting the plot
ax_error.set_xlabel('Time (s)')
ax_error.set_ylabel('Error (Euclidean Distance)')
ax_error.set_title(f'Average Error of Predicted Trajectories over Time\n'
                    f'Obs Horizon: {obs_horizon}, Pred Horizon: {pred_horizon}, Step Size: {step_size*1/50} seconds')
ax_error.legend()
plt.show()





