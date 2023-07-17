
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
<<<<<<< HEAD
    path_hyperparams = './tb_logs/version_619/hparams.yaml'
    path_checkpoint = './tb_logs/version_619/checkpoints/epoch=46.ckpt'
    filepath = './tb_logs/version_619/STATS.pkl'
=======
    # ? path_hyperparams = './tb_logs/version_588/hparams.yaml'
    # ? path_checkpoint = './tb_logs/version_588/checkpoints/epoch=55.ckpt'
    
    path_hyperparams = './tb_logs/version_607/hparams.yaml'
    path_checkpoint = './tb_logs/version_607/checkpoints/epoch=26.ckpt'
    
>>>>>>> main
    dataset_name = '2023-07-15-1711_dataset_1_episodes_2_modes.zarr.zip'

    model = Diffusion_DDPM.load_from_checkpoint(
        path_checkpoint,
        hparams_file=path_hyperparams,
    )
    model.eval() 

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

    # =========== Dataloader ===========
    # Dataset dir and filename
    dataset_dir = './data'
    dataset = CarRacingDataModule(batch_size, dataset_dir, obs_horizon, pred_horizon, seed=2, stats=stats)
    dataset.setup(name=dataset_name)
    test_dataloader = dataset.val_dataloader()

    # ---- get a batch of data ----
    batch = next(iter(test_dataloader))
    x_0_predicted, _, _ = model.sample(batch=batch[0], mode='validation')
    
    translation_vector = batch[1].squeeze().cpu().detach().numpy()
   
    nPositions = batch[0]['position'].squeeze().cpu().detach().numpy()
    nActions = batch[0]['action'].squeeze().cpu().detach().numpy()
    nVelocity = batch[0]['velocity'].squeeze().cpu().detach().numpy()
    nPositionPred = x_0_predicted.squeeze()[:, :2].cpu().detach().numpy()
    nActionPred = x_0_predicted.squeeze()[:, 2:].cpu().detach().numpy()
    

    # inpainting_points = unnormalize_data( nPositions + translation_vector, stats=pos_stats) 
    positions_prediction = unnormalize_data( nPositionPred + translation_vector, stats=pos_stats) 
    actions_prediction = unnormalize_data( nActionPred , stats=action_stats)
    positions_groundtruth = unnormalize_data( nPositions  + translation_vector, stats=pos_stats)
    actions_groundtruth = unnormalize_data( nActions, stats=action_stats)

    vel0 = unnormalize_data( nVelocity[obs_horizon, :], stats=velocity_stats)
    pos0 = positions_groundtruth[obs_horizon , ...]
    
    print("pos0: ", pos0)
    print("vel0: ", vel0)
    
    # ===========  GymWrapper  ===========
    env = EnvWrapper()
    env.seed(42)
    pos_history = []
    env.reset_car(pos0[0], pos0[1], vel0[0], vel0[1])
    position_from_saved_actions = []
    for i in range(positions_prediction.shape[0]):
        action = actions_groundtruth[i, :]
        _,_,_,info = env.step(action) #env.step_noRender(actions[i, :])
        position_from_saved_actions.append( info['car_position_vector'].copy())
        env.render()
    position_from_saved_actions = np.array(position_from_saved_actions)
    env.close()

    env = EnvWrapper()
    env.seed(42)
    pos_history = []
    env.reset_car(pos0[0], pos0[1], vel0[0], vel0[1])
    for i in range(positions_prediction.shape[0]):
        action = actions_prediction[i, :]
        _,_,_,info = env.step(action) #env.step_noRender(actions[i, :])
        pos_history.append( info['car_position_vector'].copy())
        env.render()
        # print(info['car_velocity_vector'])
        # print("Linear Velocity: ", np.linalg.norm(info['car_velocity_vector']))
        # print(action)
        print( "Action: ", action , "Velocity: ", info['car_velocity_vector'], "Position: ", info['car_position_vector'] )

    pos_history = np.array(pos_history)
    env.close()

    # ===========  Plotting  ===========
    fig = plt.figure()
    fig.clf()

    # Scatter plot for 'Groundtruth'
    plt.scatter(positions_groundtruth[:, 0], positions_groundtruth[:, 1], c='b', label='Groundtruth', s = 10)
    # Scatter plot for 'Predicted by diffusion'
    plt.scatter(positions_prediction[:, 0], positions_prediction[:, 1], c='y', s = 20, label='Predicted by diffusion')
    
    plt.scatter(position_from_saved_actions[:, 0], position_from_saved_actions[:, 1], c='g', s = 10, label='Saved actions played out')
    # Add inpainted points
    # plt.scatter(inpainting_points[:, 0], inpainting_points[:, 1], c='r', label='Inpainted points', s = 10, marker='x')
    # Scatter plot for 'Predicted actions played out'
    plt.scatter(pos_history[:, 0], pos_history[:, 1], c='r', s = 10, label='Predicted actions played out')
    # Mark start position
    # Add legend
    plt.legend()
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()