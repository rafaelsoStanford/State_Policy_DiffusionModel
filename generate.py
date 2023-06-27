import argparse
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint

from models.diffusion import *
from load_data import *

import zarr
import numpy as np
import torch
import matplotlib.pyplot as plt

# =========== parser function ===========
def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    parser.add_argument('--obs_horizon', type=int, default=40, help='Observation horizon')
    parser.add_argument('--pred_horizon', type=int, default=40, help='Prediction horizon')
    parser.add_argument('--action_horizon', type=int, default=1, help='Action horizon')
    parser.add_argument('--inpaint_horizon', type=int, default=5, help='Inpaining horizon, which denotes the amount of steps of our observations to use for inpainting')
    parser.add_argument('--noise_steps', type=int, default=1000, help='Denoising steps')
    
    parser.add_argument('--cond_dim', type=int, default=128+2+3, help='Dimension of diffusion input state')
    parser.add_argument('--output_dim', type=int, default=5, help='Dimension of diffusion output state')
    parser.add_argument('--model', type=str, default='default', help='String for choosing model architecture')

    parser.add_argument('--dataset_dir', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--dataset', type=str, default='ThreeBehaviours_20Eps.zarr.zip', help='zarr.zip dataset filename')
    
    return parser.parse_args()

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

def main(args):

    # ===========Parameters===========
    # training parameters
    n_epochs = args.n_epochs
    AMP = args.amp
    batch_size = args.batch_size
    lr = args.lr

    # diffusion parameters
    noise_steps = args.noise_steps
    obs_horizon = args.obs_horizon
    pred_horizon = args.pred_horizon
    inpaint_horizon = args.inpaint_horizon
    action_horizon = args.action_horizon
    cond_dim = args.cond_dim
    output_dim = args.output_dim


    # Dataset dir and filename
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset

    # model architecture
    model = args.model

    model = Diffusion.load_from_checkpoint(
        checkpoint_path="./tb_logs/version_389/checkpoints/epoch=24.ckpt",
    )
    model.eval()



    dataset = CarRacingDataModule (dataset_dir , batch_size, obs_horizon, pred_horizon ,action_horizon)
    dataset.setup()
    val_dataloader = dataset.val_dataloader()




    # Get batch random images from the validation set
    batch = next(iter(val_dataloader))
    batch = batch.to(model.device)

    with torch.no_grad():
        embeddings = encoder(batch)
        print("⚡" * 20, "\nPredictions (batch image embeddings):\n", embeddings.shape, "\n", "⚡" * 20)
        reconstructions = decoder(embeddings)

        # Reshape the images to (batch_size, height, width, channels)
        images_orig = batch.permute(0, 2, 3, 1).cpu().detach().numpy()
        images = reconstructions.permute(0, 2, 3, 1).cpu().detach().numpy()

        # Create a figure and axis
        #fig, ax = plt.subplots(1, 1, figsize=(16, 3))

        # Set axis labels and title
        
        # Display first image
        fig, axs = plt.subplots(1, 2)

        fig = plt.figure(figsize=(20, 20))
        columns = 8
        rows = 2
        for i in range(1, columns*1 +1):
            img = images_orig[i, :]
            fig.add_subplot(1, columns, i)
            plt.imshow(img)

        for i in range(1, columns +1):
            img = images[i, :]
            fig.add_subplot(2, columns, i)
            plt.imshow(img)
        # Show the plot
        plt.show()