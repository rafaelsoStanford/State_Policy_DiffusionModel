import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import pickle
import zarr
import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint

from models.diffusion import *



# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def save_stats(stats, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(stats, f)

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx # All index in one episode

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result



# =========== data set class  ===========
# loading data from zarr file
class CarRacingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):
        
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.sequence_len = obs_horizon + pred_horizon # chunk lenght of data
        self.normalized_train_data = {}

        

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')

        # float32, [0,1], (N,96,96,3)
        train_image_data = dataset_root['data']['img'][:]
        train_image_data = np.moveaxis(train_image_data, -1,1)
        # (N,3,96,96) Meaning N images, 3 channels, 96x96 pixels
        #train_actions_data = dataset_root['data']['action'][:] # (N,3)

        # (N, D)
        train_data = {
            # Create Prediction Targets
            'position': dataset_root['data']['position'][:], # (T,2)
#            'velocities_pred': dataset_root['data']['velocity'][:] # (T,2)
            'action': dataset_root['data']['action'][:] #(T,3)
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]


        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length= obs_horizon+pred_horizon,
            pad_before= 0,
            pad_after= 0)

        # ========== Normalize Data ============ 
        # normalized data to [-1,1], images are assumed to be normalized 
        # stats = dict()
        # normalized_train_data = dict()
        # for key, data in train_data.items():
        #     stats[key] = get_data_stats(data)
        #     normalized_train_data[key] = normalize_data(data, stats[key])
        # self.stats = stats

        self.normalized_train_data['image'] = train_image_data[:,: , :80, :80] # Assumed to be already normalized, cropped to 80x80 removing the black border
        self.normalized_train_data['action'] = train_data['action'] # All action space values are constrained to [-1,1]
        self.normalized_train_data['position'] = train_data['position'] # normalized_train_data ['position'] 
        self.indices = indices
        # self.stats = stats
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=    self.sequence_len,
            buffer_start_idx=   buffer_start_idx,
            buffer_end_idx=     buffer_end_idx,
            sample_start_idx=   sample_start_idx,
            sample_end_idx=     sample_end_idx
        )

        # # ========== normalize sample ============
        sample_stat = get_data_stats(nsample['position'])
        sample_normalized = normalize_data(nsample['position'], sample_stat)
        translation_vec = sample_normalized[0,:]
        nsample_centered = sample_normalized - translation_vec
        nsample['position'] = nsample_centered / 2.0

        # position_sample = nsample['position']
        # translation_vec = position_sample[0,:]
        # position_sample_centered = position_sample - translation_vec
        # stats = get_data_stats(position_sample_centered)
        # position_sample_normalized = normalize_data(position_sample_centered, stats)
        # nsample['position'] = position_sample_normalized 

        

        return nsample
        









# =========== data module ===========
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
        self.data_full = CarRacingDataset(  dataset_path= os.path.join(self.data_dir, name),
                                            pred_horizon=self.T_pred,
                                            obs_horizon=self.T_obs,
                                            action_horizon=self.T_act)

        self.data_train, self.data_val = random_split(self.data_full, [int(len(self.data_full)*0.8), len(self.data_full) - int(len(self.data_full)*0.8)])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=2)



############################
#========== MAIN ===========
############################

def main(args):

    # Parameters:
    n_epochs = args.n_epochs
    AMP = args.amp
    batch_size = args.batch_size
    obs_horizon = args.obs_horizon
    pred_horizon = args.pred_horizon
    action_horizon = args.action_horizon
    cond_dim = args.cond_dim
    output_dim = args.output_dim
    model = args.model


    # ===========data===========
    # Load Dataset using Pytorch Lightning DataModule
    dataset = CarRacingDataModule("./data" , batch_size, obs_horizon, pred_horizon ,action_horizon)
    
    dataset.setup(name='multipleDrivingBehaviours_testing_20eps_normalized.zarr.zip')
    train_dataloader = dataset.train_dataloader()
    valid_dataloader = dataset.val_dataloader()

    batch = next(iter(train_dataloader))

    print("Visualizing Batch and Data structure")
    print(" [ B, t_sequence, dims ]")
    for key, value in batch.items():
        print()
        print(f'--> Key: {key}')
        print(f'Shape: ({len(value)}, {len(value[0])})')
        print(value.shape)
        print("Min: ", value.min())
        print("Max: ", value.max())
        print()
    
    # for traj in range(batch['position'].shape[0]):
    #     plt.plot(batch['position'][traj,:,0], batch['position'][traj,:,1])
    #     plt.scatter(0,0, c='r')
    #     plt.waitforbuttonpress()
    #     plt.close()


    # # ===========model===========
    diffusion = Diffusion(
                    noise_steps= 1000,
                    obs_horizon=obs_horizon,
                    pred_horizon= pred_horizon,
                    observation_dim=cond_dim,
                    prediction_dim= output_dim,
                    model=model,
                    )
    # -----PL configs-----
    tensorboard = pl_loggers.TensorBoardLogger(save_dir="tb_logs/",name='',flush_secs=1)

    early_stop_callback = EarlyStopping(monitor='lr', stopping_threshold=2e-6, patience=n_epochs)   
    checkpoint_callback = ModelCheckpoint(filename="{epoch}",     # Checkpoint filename format
                                          save_top_k=-1,          # Save all checkpoints
                                          every_n_epochs=5,               # Save every epoch
                                          save_on_train_epoch_end=True,
                                          verbose=True)
    # train model
    trainer = pl.Trainer(accelerator='gpu', devices=[0,1], precision=("16-mixed" if AMP else 32), max_epochs=n_epochs, 
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=tensorboard, profiler="simple", val_check_interval=0.25, 
                         accumulate_grad_batches=1, gradient_clip_val=0.5 ) #, strategy='ddp_find_unused_parameters_true')

    trainer.validate(model= diffusion, dataloaders=valid_dataloader)
    trainer.fit(model=diffusion, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    parser.add_argument('--obs_horizon', type=int, default=40, help='Observation horizon')
    parser.add_argument('--pred_horizon', type=int, default=40, help='Prediction horizon')
    parser.add_argument('--action_horizon', type=int, default=1, help='Action horizon')
    
    parser.add_argument('--cond_dim', type=int, default=514, help='Dimension of diffusion input state')
    parser.add_argument('--output_dim', type=int, default=2, help='Dimension of diffusion output state')
    parser.add_argument('--model', type=str, default='default', help='String for choosing model architecture')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)