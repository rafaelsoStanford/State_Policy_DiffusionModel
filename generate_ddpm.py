
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
        denosing_steps = 250
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
    model.sample(batch=batch, mode='test')

    # # =========== Pytorch Lightning Trainer  ===========
    # trainer = pl.Trainer(accelerator='gpu', devices=[0], precision=("16-mixed" if AMP else 32), max_epochs=n_epochs)
    # trainer.test(model, dataloaders=test_dataloaders)

if __name__ == "__main__":
    main()
    