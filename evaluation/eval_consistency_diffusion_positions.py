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

def plot_predictions(position_prediction_history, position_groundtruth, obs_horizon,  pred_horizon, step_size, inpaint_horizon = 0):
    """Plots the position predictions"""
    fig, ax = plt.subplots()
    for i in range(len(position_prediction_history)):
        plt.plot(position_prediction_history[i][:, 0], position_prediction_history[i][:, 1], color='green')
        plt.scatter(position_prediction_history[i][:, 0], position_prediction_history[i][:, 1], color='green' , s=10)
    plt.plot(position_groundtruth[0, :, 0], position_groundtruth[0, :, 1])
    plt.plot(position_groundtruth[0, :obs_horizon, 0], position_groundtruth[0, :obs_horizon, 1], 'o', color='red')
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
        error = np.sqrt(np.sum((position_prediction_history[i][inpaint_horizon:] - position_groundtruth[0, inpaint_horizon:, :2])**2, axis=1))
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




# GLOBAL VARIABLES
NUM_RUNS = 10
DATASET_SEED = 42
EXPERIMENT_NAME = 'Consistency evaluations' 
MODEL_NAME = 'DDPM'
STEPS = 100
DDPM_ADDITIONAL_STEPS = 200

# Paths
# # paths
dataset_path = './evaluation/data/'
dataset_name = 'EvaluationMiddle_dataset_1_episodes_1_modes.zarr.zip'

stats_file_path = './tb_logs/version_671/STATS.pkl'
checkpoint_path = './tb_logs/version_671/checkpoints/epoch=9.ckpt'
hyperparams_path = './tb_logs/version_671/hparams.yaml'

# Load the stats from the file
with open(stats_file_path, 'rb') as stats_file:
    stats = pickle.load(stats_file)
stats = stats[0]

# Load the model
model = load_model(MODEL_NAME,  checkpoint_path, hyperparams_path)

# Fetch model parameters
model_params = fetch_hyperparams_from_yaml(hyperparams_path)
obs_horizon = model_params['obs_horizon']
pred_horizon = model_params['pred_horizon']
step_size = model_params['step_size']

# Load the dataset
dataset = CarRacingDataModule(1 , dataset_path, obs_horizon, pred_horizon, seed=5, stats=stats)
dataset.setup(name=dataset_name)
test_dataloaders = dataset.val_dataloader()
batch, translation, start_idx, end_idx = next(iter(test_dataloaders))


# Print hyperparameter information
print('Hyperparameters:')
print('Model: ', MODEL_NAME)
print('Step size: ', step_size , 'equivalent to ', step_size*1/50, 'seconds')
print('Obs horizon: ', obs_horizon , " equivalent to ", step_size*obs_horizon*1/50, "seconds")
print('Pred horizon: ', pred_horizon , " equivalent to ",  step_size*pred_horizon*1/50, "seconds")
print('Dataset: ', dataset_name)

# Position estimations
translation = translation.cpu().detach().numpy()
position_groundtruth = unnormalize_position(batch['position'], translation, stats['position'])
position_prediction_history = [] 

print("*** Sampling for position estimations***")
for i in range(NUM_RUNS):
    print("*** Sample from model ***")
    if MODEL_NAME == 'DDPM':
        predicted_sample, _, _ = model.sample(batch=batch, mode='validation')
    elif MODEL_NAME == 'DDIM':
        predicted_sample, _, _ = model.sample(batch=batch, mode='validation', step_size=STEPS)
    elif MODEL_NAME == 'DDIPM':
        predicted_sample, _, _ = model.sample(batch=batch, mode='validation', step_size=STEPS, ddpm_steps=DDPM_ADDITIONAL_STEPS)

    print("*** Unnormalize sample ***")
    predicted_sample = predicted_sample.squeeze().cpu().detach().numpy()
    nPosition_predicted = predicted_sample[:, 0:2]
    position_pred = unnormalize_position(nPosition_predicted , translation, stats['position']) 
    position_prediction_history.append(position_pred.copy())

# Plotting
plot_predictions(position_prediction_history, position_groundtruth, obs_horizon, pred_horizon, step_size, 10)










# import sys
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt

# # setting path
# sys.path.append('../diffusion_bare')

# from envs.car_racing import CarRacing
# from models.diffusion_ddpm import *
# from models.diffusion_ddim import *
# from utils.load_data import *
# from utils.data_utils import *
# from generateData.trajectory_control_utils import *


# # ###### GLOBAL VARIABLES ######
# NUM_RUNS = 2
# DATASET_SEED = 42
# EXPERIMENT_NAME = 'Consistency evaluations' 
# MODEL_NAME = 'DDPM'

# # DDIM hyperparameters
# STEPS = 20
# DDPM_ADDITIONAL_STEPS = 100

# def load_model(model_name, checkpoint_path, hparams_path):
#     if model_name == 'DDPM':
#         model = Diffusion_DDPM.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path)
#     elif model_name == 'DDIM' or model_name == 'DDIPM':
#         model = Diffusion_DDIM.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path)
#     else:
#         raise ValueError("Model name must be either 'DDPM', 'DDIM', or 'DDIPM'")
#     model.eval()
#     return model

# # # paths
# dataset_path = './evaluation/data/'
# dataset_name = 'EvaluationRight_dataset_1_episodes_1_modes.zarr.zip'

# filepath = './tb_logs/version_667/STATS.pkl'
# path_checkpoint = './tb_logs/version_667/checkpoints/epoch=12.ckpt'
# path_hyperparams = './tb_logs/version_667/hparams.yaml'

# with open(filepath, 'rb') as f:
#     stats = pickle.load(f)
# stats = stats[0]

# #Model
# model = load_model(MODEL_NAME,  path_checkpoint, path_hyperparams)
# model_params = fetch_hyperparams_from_yaml(path_hyperparams)
# obs_horizon = model_params['obs_horizon']
# pred_horizon = model_params['pred_horizon']
# # Dataloader
# dataset = CarRacingDataModule(1 , dataset_path, obs_horizon, pred_horizon, seed=DATASET_SEED, stats=stats)
# dataset.setup(name=dataset_name)
# test_dataloaders = dataset.val_dataloader()
# batch, translation, start_idx, end_idx = next(iter(test_dataloaders))
# step_size = model_params['step_size']

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

#  # ==================================================== #
# print("*** Sampling for position estimations***")
# for i in range(NUM_RUNS):
#     print()
#     print("*** Sample from model ***")
#     if MODEL_NAME == 'DDPM':
#         x_0_predicted, _, _ = model.sample(batch=batch, mode='validation')
#     elif MODEL_NAME == 'DDIM':
#         x_0_predicted, _, _ = model.sample(batch=batch, mode='validation', step_size=STEPS)
#     elif MODEL_NAME == 'DDIPM':
#         x_0_predicted, _, _ = model.sample(batch=batch, mode='validation', step_size=STEPS, ddpm_steps=DDPM_ADDITIONAL_STEPS)
#     print("      ... Sampled ***")

#     print("*** Unnormalize sample ***")
#     x_0_predicted = x_0_predicted.squeeze().cpu().detach().numpy()
#     nPosition_predicted = x_0_predicted[:, 0:2]
#     position_pred = unnormalize_position(nPosition_predicted , translation, stats['position']) 
#     position_prediction_history.append(position_pred.copy())
#     print("     ...Unnormalized ***")
# print("*** Plotting ***")

# fig, ax = plt.subplots()
# for i in range(NUM_RUNS):
#     plt.plot(position_prediction_history[i][:, 0], position_prediction_history[i][:, 1], color='green')
#     plt.scatter(position_prediction_history[i][:, 0], position_prediction_history[i][:, 1], color='green' , s=10)
# plt.plot(position_groundtruth[0, :, 0], position_groundtruth[0, :, 1])
# plt.plot(position_groundtruth[0, :obs_horizon, 0], position_groundtruth[0, :obs_horizon, 1], 'o', color='red')
# plt.title('Diffusion position predicitons, with ' + MODEL_NAME + ' model, \n Horizons: ' + str(obs_horizon) + ' obs, ' + str(pred_horizon) + ' pred, Step size: ' + str(step_size*1/50) + ' seconds')
# # Equal aspect ratio
# ax.set_aspect('equal', 'box')
# plt.show()

