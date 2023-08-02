import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

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


# GLOBAL VARIABLES
NUM_RUNS = 10
DATASET_SEED = 42
EXPERIMENT_NAME = 'Consistency evaluations' 
MODEL_NAME = 'DDIM'
STEPS = 100
DDPM_ADDITIONAL_STEPS = 200

# Paths
# # paths
dataset_path = './evaluation/data/'
dataset_name = 'Evaluation_dataset_1_episodes_1_modes.zarr.zip'

stats_file_path = './tb_logs/version_669/STATS.pkl'
checkpoint_path = './tb_logs/version_669/checkpoints/epoch=10.ckpt'
hyperparams_path = './tb_logs/version_669/hparams.yaml'

# Load the stats from the file
with open(stats_file_path, 'rb') as stats_file:
    stats = pickle.load(stats_file)
stats = stats[0]

print("*** Loading Data ...")
dataset_root = zarr.open(dataset_path + dataset_name, 'r')
data = {
    'image': np.moveaxis(dataset_root['data']['img'][:], -1, 1),
    'position': dataset_root['data']['position'][:],
    'velocity': dataset_root['data']['velocity'][:],
    'action': dataset_root['data']['action'][:],
}
model_params = fetch_hyperparams_from_yaml(hyperparams_path)
obs_horizon = model_params['obs_horizon']
pred_horizon = model_params['pred_horizon']
inpaint_horizon = model_params['inpaint_horizon']
step_size = model_params['step_size']


with open(stats_file_path, 'rb') as f:
    stats = pickle.load(f)[0]

# -----  Loading and initializing model ----- #
model = Diffusion_DDPM.load_from_checkpoint(
    checkpoint_path,
    hparams_file=hyperparams_path,
)
model.eval()



sequence_len = obs_horizon + pred_horizon
indices = create_sample_indices_sparse(
    ends= [data['position'].shape[0]],
    sequence_length= sequence_len,
    step_size= step_size)


error_sequence_list = []
for j in range(len(indices)):

    start_idx, end_idx, _, _ =  indices[j]  # indices[np.random.randint(0, len(indices))]
    sample = sample_sequence_sparse(
        data= data,
        step_size= step_size,
        sample_start_idx= start_idx,
        sample_end_idx= end_idx,
    )

    # Normalize the data
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

    batch = {
        'position': torch.from_numpy(nsample['position']).unsqueeze(0).float(),
        'velocity': torch.from_numpy(nsample['velocity']).unsqueeze(0).float(),
        'action': torch.from_numpy(nsample['action']).unsqueeze(0).float(),
        'image': torch.from_numpy(nsample['image']).unsqueeze(0).float(),
    }

    # Print hyperparameter information
    print('Hyperparameters:')
    print('Model: ', MODEL_NAME)
    print('Step size: ', step_size , 'equivalent to ', step_size*1/50, 'seconds')
    print('Obs horizon: ', obs_horizon , " equivalent to ", step_size*obs_horizon*1/50, "seconds")
    print('Pred horizon: ', pred_horizon , " equivalent to ",  step_size*pred_horizon*1/50, "seconds")
    print('Dataset: ', dataset_name)

    # Position estimations
    # translation = Translation.cpu().detach().numpy()
    position_groundtruth = unnormalize_position(batch['position'], Translation, stats['position'])
    position_prediction_history = [] 

    predicted_sample, _, _ = model.sample(batch=batch, mode='validation')


    predicted_sample = predicted_sample.squeeze().cpu().detach().numpy()
    nPosition_predicted = predicted_sample[:, 0:2]
    position_pred = unnormalize_position(nPosition_predicted , Translation, stats['position']) 
    position_prediction_history.append(position_pred.copy())

    # Compute the error
    error_sequence = np.linalg.norm(position_groundtruth[0, obs_horizon:, :] - position_pred[inpaint_horizon:], axis=1)
    error_sequence_list.append(error_sequence)

# Compute the mean error for each entry over all the sequences
mean_error = np.mean(np.array(error_sequence_list), axis=0)
print("Mean error: ", mean_error)

plt.figure()
plt.plot(mean_error)
plt.xlabel('Time steps')
plt.ylabel('Mean error')
plt.title('Mean error over time')
plt.show()

