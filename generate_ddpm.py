
import matplotlib.pyplot as plt
import yaml
import pickle
import time
import cv2

from models.diffusion_ddpm import *
from models.diffusion_ddim import *
from utils.load_data import *
from envs.envWrapper import EnvWrapper

def fetch_hyperparams_from_yaml(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams


# TODO: This code needs to be cleaned up.


############################
#========== MAIN ===========
############################

def main():
    # Some params, dont know where to put them
    AMP = True
    n_epochs = 1
    batch_size = 1

    # =========== Load Model ===========
    # ? path_hyperparams = './tb_logs/version_588/hparams.yaml'
    # ? path_checkpoint = './tb_logs/version_588/checkpoints/epoch=55.ckpt'
    
    path_hyperparams = './tb_logs/version_590/hparams.yaml'
    path_checkpoint = './tb_logs/version_590/checkpoints/epoch=17.ckpt'
    
    dataset_name = '2023-07-12-2225_dataset_1_episodes_3_modes.zarr.zip'

    model = Diffusion_DDPM.load_from_checkpoint( #Choose between DDPM and DDIM -- Model is inherited from DDPM thus they should be compatible
        path_checkpoint,
        hparams_file=path_hyperparams,
    )
    model.eval() 
    
    # Specify the path to the pickle file
    filepath = 'MinMax.pkl'
    # Load the pickle file
    with open(filepath, 'rb') as f:
        stats = pickle.load(f)

    pos_stats = stats[0]['position']
    action_stats = stats[0]['action']
    velocity_stats = stats[0]['velocity']

    # ===========Parameters===========
    model_params = fetch_hyperparams_from_yaml(path_hyperparams)
    obs_horizon = model_params['obs_horizon']
    pred_horizon = model_params['pred_horizon']
    inpaint_horizon = model_params['inpaint_horizon']

    # =========== Dataloader ===========
    # Dataset dir and filename
    dataset_dir = './data'
    dataset = CarRacingDataModule_forInference( batch_size, dataset_dir, obs_horizon, pred_horizon ,seed=52, stats=stats)
    dataset.setup( name = dataset_name )
    test_dataloaders = dataset.val_dataloader()

    # TODO: We only use a single batch, which begs the question on why using a dataloader in the first place.
    # ---- get a batch of data ----
    batch = next(iter(test_dataloaders))
    x_0_predicted , _ , _ = model.sample(batch=batch[0], mode='validation')
    
    # ---- seperate position / action ----
    # TODO: Very dirty, should be cleaned up
    output = x_0_predicted.squeeze().cpu().detach().numpy()
    
    inpaint = unnormalize_data(2* output[:inpaint_horizon, :2] + batch[1].detach().cpu().numpy() , stats= pos_stats)
    prediction = output[inpaint_horizon:, :]
    
    prediction_position = unnormalize_data( 2* prediction[:, :2] + batch[1].detach().cpu().numpy() , stats= pos_stats)
    actions = unnormalize_data(prediction[:, 2:], stats= action_stats)
    
    position_groundtruth = unnormalize_data(2 * batch[0]['position'].squeeze() + batch[1].detach().cpu().numpy(), stats= pos_stats)
    vel0 = unnormalize_data(batch[0]['velocity'].squeeze()[obs_horizon,:] , stats= velocity_stats)
    pos0 = inpaint[-1, :2]
    
    
    # ===========  GymWrapper  ===========
    env = EnvWrapper()
    env.seed(42)
    pos_history = []
    pos_history.append(pos0)
    # env.reset()
    # heading vector 
    #direction = np.array(inpaint[-2, :2]) - np.array(inpaint[-1, :2])
    env.reset_car(pos0[0], pos0[1], vel0[0], vel0[1])
    for i in range(prediction_position.shape[0]):
        print("Action: ", actions[i, :])
        action = actions[i, :]
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], 0, 1)
        action[2] = np.clip(action[2], 0, 1)
        _,_,_,info = env.step(actions[i, :]) #env.step_noRender(actions[i, :])
        pos_history.append( info['car_position_vector'].copy())
        env.render()
        print(info['car_velocity_vector'])
        input("Press Enter to continue...")
    pos_history = np.array(pos_history)
    env.close()

    # ===========  Plotting  ===========
    # TODO: Plotting should be moved to a seperate file
    fig = plt.figure()
    fig.clf()
    # Create a colormap for fading colors based on the number of timesteps
    cmap = get_cmap('viridis', pred_horizon )
    cmap2 = get_cmap('Reds', pred_horizon )
    
    # Create an array of indices from 0 to timesteps-1
    indices = np.arange(pred_horizon )
    # Normalize the indices to the range [0, 1]
    normalized_indices = indices / (pred_horizon  - 1)
    # Create a color array using the colormap and normalized indices
    colors = cmap(normalized_indices)
    colors2 = cmap2(normalized_indices)
    
    
    # Scatter plot for 'Groundtruth'
    plt.scatter(position_groundtruth[:, 0], position_groundtruth[:, 1], c='b', label='Groundtruth', s = 20)
    # Scatter plot for 'Predicted by diffusion'
    plt.scatter(prediction_position[:, 0], prediction_position[:, 1], c=colors, s = 20, label='Predicted by diffusion')
    
    
    # Add inpainted points
    plt.scatter(inpaint[:, 0], inpaint[:, 1], c='k', label='Inpainted points', s = 20)
    # Scatter plot for 'Predicted actions played out'
    # plt.scatter(pos_history[:, 0], pos_history[:, 1], c='r', label='Predicted actions played out', s = 20)
    plt.scatter(pos_history[1:, 0], pos_history[1:, 1], c=colors2, s = 10, label='Predicted actions played out')
    # Mark start position
    plt.scatter(pos0[0], pos0[1], c='g', label='Start position', s = 10)
    # Draw velocity vector at start position
    plt.arrow(pos0[0], pos0[1], vel0[0], vel0[1], width=0.01, color='g', label='Start velocity')
    # Add legend
    plt.legend()
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()