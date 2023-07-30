
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint

from models.diffusion_ddpm import *
from models.diffusion_ddim import *
from utils.load_data import *

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
    batch_size = 1

    # =========== Load Model ===========
    # =========== Load Model ===========
    # path_hyperparams = './tb_logs/version_643/hparams.yaml'
    # path_checkpoint = './tb_logs/version_643/checkpoints/epoch=32.ckpt'
    # filepath = './tb_logs/version_643/STATS.pkl'
    # dataset_name = '2023-07-17-2252_dataset_1_episodes_2_modes.zarr.zip'
    path_hyperparams = './tb_logs/version_659/hparams.yaml'
    path_checkpoint = './tb_logs/version_659/checkpoints/epoch=12.ckpt'
    filepath = './tb_logs/version_659/STATS.pkl'
    dataset_name = '2023-07-17-2252_dataset_1_episodes_2_modes.zarr.zip'
    # Load the pickle file
    with open(filepath, 'rb') as f:
        stats = pickle.load(f)
    stats = stats[0]

    model = Diffusion_DDPM.load_from_checkpoint( #Choose between DDPM and DDIM -- Model is inherited from DDPM thus they should be compatible
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
    dataset = CarRacingDataModule( batch_size, dataset_dir, obs_horizon, pred_horizon ,seed=125 , stats=stats)

    dataset.setup( name = dataset_name )
    test_dataloaders = dataset.val_dataloader()

    batch = next(iter(test_dataloaders))
    sampling_history =  model.sample(batch=batch[0], mode='test')

    # =========== Save ===========
    plt_toVideo(model,
                sampling_history,
                batch)

if __name__ == "__main__":
    main()
