
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
    batch_size = 1

    # =========== Load Model ===========
    path_hyperparams = './tb_logs/version_624/hparams.yaml'
    path_checkpoint = './tb_logs/version_624/checkpoints/epoch=35.ckpt'
    filepath = './tb_logs/version_624/STATS.pkl'
    #dataset_name = '2023-07-15-1711_dataset_1_episodes_2_modes.zarr.zip'
    dataset_name = '2023-07-17-2252_dataset_1_episodes_2_modes.zarr.zip'

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
    inpaint_horizon = model_params['inpaint_horizon']

    # =========== Dataloader ===========
    # Dataset dir and filename
    dataset_dir = './data'
    dataset = CarRacingDataModule(batch_size, dataset_dir, obs_horizon, pred_horizon, seed=156, stats=stats)
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
    positions_prediction = unnormalize_data( 2* nPositionPred + translation_vector, stats=pos_stats) 
    actions_prediction = unnormalize_data( nActionPred , stats=action_stats)
    positions_groundtruth = unnormalize_data( 2* nPositions  + translation_vector, stats=pos_stats)
    actions_groundtruth = unnormalize_data( nActions, stats=action_stats)

    vel0 = unnormalize_data( nVelocity[inpaint_horizon-1, ...], stats=velocity_stats)
    pos0 = positions_groundtruth[inpaint_horizon-1 , ...]
    
    print("pos0: ", pos0)
    print("vel0: ", vel0)


    initAngle = batch[0]['angle'].squeeze()[obs_horizon].cpu().detach().numpy()
    
    # ===========  GymWrapper  ===========
    env = EnvWrapper()
    env.seed(42)
    pos_history = []
    env.reset_car(pos0[0], pos0[1], vel0[0], vel0[1], initAngle)
    position_from_saved_actions = []

    actions_gt = actions_groundtruth[inpaint_horizon:, ...].copy()
    for i in range(actions_gt.shape[0]):
        action = actions_gt[i, :]
        _,_,_,info = env.step(action) 
        position_from_saved_actions.append( info['car_position_vector'].copy())
        # env.render()
    position_from_saved_actions = np.array(position_from_saved_actions)
    env.close()

    env = EnvWrapper()
    env.seed(42)
    pos_history = []
    env.reset_car(pos0[0], pos0[1], vel0[0], vel0[1], initAngle)
    for i in range(positions_prediction.shape[0]):
        action = actions_prediction[i, :]
        _,_,_,info = env.step(action) #env.step_noRender(actions[i, :])

        pos_history.append( info['car_position_vector'].copy())

        # env.render()
        print( "Action: ", action , "Velocity: ", info['car_velocity_vector'], "Position: ", info['car_position_vector'] )

    pos_history = np.array(pos_history)
    env.close()


    # ==================================#
    # ===========  Plotting  ===========#
    # ==================================#

    fig = plt.figure()
    plt.scatter(pos0[0], pos0[1], c='k', label='Initial position', s = 50)
    plt.plot(positions_groundtruth[:, 0], positions_groundtruth[:, 1], c='b', label='Groundtruth')
    plt.scatter(positions_prediction[:, 0], positions_prediction[:, 1], c='y', s = 10, label='Predicted by diffusion')
    plt.scatter(position_from_saved_actions[:, 0], position_from_saved_actions[:, 1], c='g', s = 10, label='Saved actions played out')
    plt.scatter(pos_history[:, 0], pos_history[:, 1], c='r', s = 10, label='Predicted actions played out')
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(3, 3)
    axs[0][0].plot(nActionPred[:,0], label='Predicted normalized')
    axs[0][0].plot(nActions[inpaint_horizon:,0], label='Groundtruth normalized')
    legend1 = axs[0][0].legend(fontsize='small', loc='upper left')

    axs[1][0].plot(actions_prediction[:,0], label='Predicted unnormalized')
    axs[1][0].plot(actions_groundtruth[inpaint_horizon:,0], label='Groundtruth unnormalized')
    legend2 = axs[1][0].legend(fontsize='small', loc='upper left')

    axs[0][1].plot(nActionPred[:,1], label='Predicted normalized')
    axs[0][1].plot(nActions[inpaint_horizon:,1], label='Groundtruth normalized')
    legend3 = axs[0][1].legend(fontsize='small', loc='upper left')

    axs[1][1].plot(actions_prediction[:,1], label='Predicted unnormalized')
    axs[1][1].plot(actions_groundtruth[inpaint_horizon:,1], label='Groundtruth unnormalized')
    legend4 = axs[1][1].legend(fontsize='small', loc='upper left')
    
    axs[0][2].plot(nActionPred[:,2], label='Predicted normalized')
    axs[0][2].plot(nActions[inpaint_horizon:,2], label='Groundtruth normalized')
    legend5 = axs[0][2].legend(fontsize='small', loc='upper left')

    axs[1][2].plot(actions_prediction[:,2], label='Predicted unnormalized')
    axs[1][2].plot(actions_groundtruth[inpaint_horizon:,2], label='Groundtruth unnormalized')
    legend6 = axs[1][2].legend(fontsize='small', loc='upper left')

    axs[0][0].set_title('Steering', pad=30)
    axs[0][1].set_title('Accerleration / Gas', pad=30)
    axs[0][2].set_title('Beaking', pad=30)

    # Modify the size and position of the legends
    legend1.get_frame().set_alpha(0.8)  # Adjust the legend background transparency
    legend1.set_bbox_to_anchor((0.5, 1.15))  # Adjust the legend position relative to the plot
    legend2.get_frame().set_alpha(0.8)
    legend2.set_bbox_to_anchor((0.5, 1.15))
    legend3.get_frame().set_alpha(0.8)
    legend3.set_bbox_to_anchor((0.5, 1.15))
    legend4.get_frame().set_alpha(0.8)
    legend4.set_bbox_to_anchor((0.5, 1.15))
    legend5.get_frame().set_alpha(0.8)
    legend5.set_bbox_to_anchor((0.5, 1.15))
    legend6.get_frame().set_alpha(0.8)
    legend6.set_bbox_to_anchor((0.5, 1.15))

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    axs[0][0].grid(True)
    axs[1][0].grid(True)
    axs[2][0].grid(True)
    axs[0][1].grid(True)
    axs[1][1].grid(True)
    axs[2][1].grid(True)
    axs[0][2].grid(True)
    axs[1][2].grid(True)
    axs[2][2].grid(True)
    plt.show()


if __name__ == "__main__":
    main()