
import matplotlib.pyplot as plt
import yaml
import pickle


from models.diffusion_ddpm import *
from models.diffusion_ddim import *
from utils.load_data import *
from envs.envWrapper import EnvWrapper

def fetch_hyperparams_from_yaml(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams


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

    model = Diffusion_DDPM.load_from_checkpoint( #Choose between DDPM and DDIM -- Model is inherited from DDPM thus they are compatible
        path_checkpoint,
        hparams_file=path_hyperparams,
    )
    model.eval() 
    
    # Specify the path to the pickle file
    filepath = 'MinMax.pkl'
    # Load the pickle file
    with open(filepath, 'rb') as f:
        stats = pickle.load(f)

    stats = stats[0]
    pos_stats = stats['position']
    action_stats = stats['action']
    velocity_stats = stats['velocity']

    # ===========Parameters===========
    model_params = fetch_hyperparams_from_yaml(path_hyperparams)
    obs_horizon = model_params['obs_horizon']
    pred_horizon = model_params['pred_horizon']
    inpaint_horizon = model_params['inpaint_horizon']

    # =========== Dataloader ===========
    # Dataset dir and filename
    dataset_dir = './data'
    dataset = CarRacingDataModule(batch_size, dataset_dir, obs_horizon, pred_horizon, seed=52, stats=stats)
    dataset.setup(name=dataset_name)
    test_dataloader = dataset.val_dataloader()

    # ---- get a batch of data ----
    batch = next(iter(test_dataloader))
    x_0_predicted, _, _ = model.sample(batch=batch[0], mode='validation')
    translation_vector = batch[1].squeeze().cpu().detach().numpy()
    nPositions = batch[0]['position'].squeeze()

    # ---- separate position / action ----
    output = x_0_predicted.squeeze().cpu().detach().numpy()
    positions = output[:, :2]
    actions = output[:, 2:]

    inpainting_points = unnormalize_data(2 * positions[:inpaint_horizon, ...] + translation_vector, stats=pos_stats)
    positions_prediction = unnormalize_data(2 *  positions[inpaint_horizon:, ...]  + translation_vector, stats=pos_stats)
    actions_prediction = unnormalize_data(2 * actions[inpaint_horizon:, ...], stats=action_stats)
    positions_groundtruth = unnormalize_data(2 *nPositions  + translation_vector, stats=pos_stats)

    vel0 = unnormalize_data(batch[0]['velocity'].squeeze()[obs_horizon, :], stats=velocity_stats)
    pos0 = positions_groundtruth[obs_horizon , ...]
    
    print("pos0: ", pos0)
    print("vel0: ", vel0)
    
    # ===========  GymWrapper  ===========
    env = EnvWrapper()
    env.seed(42)
    pos_history = []
    env.reset_car(pos0[0], pos0[1], vel0[0], vel0[1])
    for i in range(positions_prediction.shape[0]):
        action = actions_prediction[i, :]
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], 0, 1)
        action[2] = np.clip(action[2], 0, 1)
        _,_,_,info = env.step(actions[i, :]) #env.step_noRender(actions[i, :])
        pos_history.append( info['car_position_vector'].copy())
        env.render()
        print(info['car_velocity_vector'])
        print("Linear Velocity: ", np.linalg.norm(info['car_velocity_vector']))
        print(action)
        # input("Press Enter to continue...")
    pos_history = np.array(pos_history)
    env.close()

    # ===========  Plotting  ===========
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
    colors2 = cmap2(np.flip(normalized_indices))
    # Scatter plot for 'Groundtruth'
    plt.scatter(positions_groundtruth[:, 0], positions_groundtruth[:, 1], c='b', label='Groundtruth', s = 10)
    # Scatter plot for 'Predicted by diffusion'
    plt.scatter(positions_prediction[:, 0], positions_prediction[:, 1], c=colors, s = 10, label='Predicted by diffusion')
    # Add inpainted points
    plt.scatter(inpainting_points[:, 0], inpainting_points[:, 1], c='o', label='Inpainted points', s = 10)
    # Scatter plot for 'Predicted actions played out'
    plt.scatter(pos_history[:, 0], pos_history[:, 1], c=colors2, s = 10, label='Predicted actions played out')
    # Mark start position
    # Add legend
    plt.legend()
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()