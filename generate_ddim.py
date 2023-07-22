from models.diffusion_ddpm import *
from models.diffusion_ddim import *
from utils.load_data import *


############################
#========== MAIN ===========
############################

def main():
    batch_size = 1

    # =========== Load Model ===========
    # ? path_hyperparams = './tb_logs/version_588/hparams.yaml'
    # ? path_checkpoint = './tb_logs/version_588/checkpoints/epoch=55.ckpt'
    
    path_hyperparams = './tb_logs/version_590/hparams.yaml'
    path_checkpoint = './tb_logs/version_590/checkpoints/epoch=17.ckpt'
    
    dataset_name = '2023-07-18-0031_dataset_1_episodes_2_modes.zarr.zip'

    model = Diffusion_DDIM.load_from_checkpoint( #Choose between DDPM and DDIM -- Model is inherited from DDPM thus they should be compatible
        path_checkpoint,
        hparams_file=path_hyperparams,
    )
    model.eval() 

    # ===========Parameters===========
    model_params = fetch_hyperparams_from_yaml(path_hyperparams)
    obs_horizon = model_params['obs_horizon']
    pred_horizon = model_params['pred_horizon']

    # =========== Dataloader ===========
    # Dataset dir and filename
    dataset_dir = './data'
    dataset = CarRacingDataModule( batch_size, dataset_dir, obs_horizon, pred_horizon ,seed=125)
    dataset.setup( name = dataset_name )
    test_dataloaders = dataset.val_dataloader()

    batch = next(iter(test_dataloaders))
    model.sample(batch=batch, step_size=50, ddpm_steps = 50)

if __name__ == "__main__":
    main()
    