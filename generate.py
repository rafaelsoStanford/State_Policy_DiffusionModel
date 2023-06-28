
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint

from models.diffusion import *
from load_data import *

import os
import yaml

def fetch_hyperparams_from_yaml(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams

# =========== data loader module ===========
# data module
class CarRacingDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 16 , T_obs=4, T_pred=8 , T_act =1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.T_act = T_act

        self.data_train = None
        self.data_val = None

    def setup(self, name: str = None):
        # ----- CarRacingDataset is a Dataloader file, takes care of normalization and such-----
        self.data_full = CarRacingDataset(  dataset_path= os.path.join(self.data_dir, name),
                                            pred_horizon=self.T_pred,
                                            obs_horizon=self.T_obs,
                                            action_horizon=self.T_act)
        self.data_train, self.data_val = random_split(self.data_full, [int(len(self.data_full)*0.8), len(self.data_full) - int(len(self.data_full)*0.8)])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=4)


############################
#========== MAIN ===========
############################

def main():
    # Some params, dont know where to put them
    AMP = True
    n_epochs = 1
    batch_size = 16

    # =========== Load Model ===========
    path_hyperparams = './tb_logs/version_416/hparams.yaml'
    model_params = fetch_hyperparams_from_yaml(path_hyperparams)
    model = Diffusion.load_from_checkpoint(
        './tb_logs/version_416/checkpoints/epoch=24.ckpt',
        hparams_file='./tb_logs/version_416/hparams.yaml'
    )
    model.eval() 

    # ===========Parameters===========
    noise_steps = model_params['noise_steps']
    obs_horizon = model_params['obs_horizon']
    pred_horizon = model_params['pred_horizon']

    # =========== Dataloader ===========
    # Dataset dir and filename
    dataset_dir = './data'
    dataset = CarRacingDataModule(dataset_dir , batch_size, obs_horizon, pred_horizon )
    dataset.setup( name='ThreeBehaviours_20Eps.zarr.zip' )
    test_dataloaders = dataset.val_dataloader()

    # =========== Pytorch Lightning Trainer  ===========

    trainer = pl.Trainer(accelerator='gpu', devices=[0,1], precision=("16-mixed" if AMP else 32), max_epochs=n_epochs)
    trainer.test(model, dataloaders=test_dataloaders)

if __name__ == "__main__":
    main()
    