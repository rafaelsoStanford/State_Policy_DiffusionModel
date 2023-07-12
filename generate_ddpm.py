

from models.diffusion_ddpm import *
from models.diffusion_ddim import *
from utils.load_data import *

from envs.envWrapper import EnvWrapper

import yaml

def fetch_hyperparams_from_yaml(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams


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
    
    dataset_name = '2023-07-10-1443_dataset_1_episodes_3_modes.zarr.zip'

    model = Diffusion_DDPM.load_from_checkpoint( #Choose between DDPM and DDIM -- Model is inherited from DDPM thus they should be compatible
        path_checkpoint,
        hparams_file=path_hyperparams,
    )
    model.eval() 

    # ===========Parameters===========
    model_params = fetch_hyperparams_from_yaml(path_hyperparams)
    obs_horizon = model_params['obs_horizon']
    pred_horizon = model_params['pred_horizon']
    inpaint_horizon = model_params['inpaint_horizon']

    # =========== Dataloader ===========
    # Dataset dir and filename
    dataset_dir = './data'
    dataset = CarRacingDataModule( batch_size, dataset_dir, obs_horizon, pred_horizon ,seed=125)
    dataset.setup( name = dataset_name )
    test_dataloaders = dataset.val_dataloader()

    batch = next(iter(test_dataloaders))
    output = model.sample(batch=batch, mode='test', denoising_steps=200)
    inpaint = output[:inpaint_horizon, :]
    prediction = output[inpaint_horizon:, :]
    prediction_position = prediction[:, :2]
    prediction_action = prediction[:, 2:]
    pos0 = unnormalize_data(inpaint[-1, :2], stats= dataset.stats['position'])
    
    # ===========  GymWrapper  ===========
    env = EnvWrapper()
    pos_history = []
    env.reset_car(pos0[0], pos0[1])
    for i in range(prediction.shape[0]):
        info = env.step_noRender(prediction_action[i, :])
        pos_history.append( info['car_position_vector'].copy() )
    pos_history = np.array(pos_history)
    env.close()

    # ===========  Plotting  ===========
    import matplotlib.pyplot as plt
    plt.plot(pos_history[:, 0], pos_history[:, 1], 'r')
    #plt.plot(batch['position'].squeeze()[:, 0], batch['position'].squeeze()[:, 1], 'b')
    #plt.plot(prediction_position[:, 0], prediction_position[:, 1], 'g')
    plt.show()

if __name__ == "__main__":
    main()
    