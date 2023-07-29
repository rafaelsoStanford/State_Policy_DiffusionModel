import numpy as np
import os
import pickle
import zarr
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from utils.data_utils import *

class CarRacingDataset(torch.utils.data.Dataset):
    """
    CarRacingDataset class for loading and normalizing data, creating datasets, and creating dataloaders.
    """
    
    def __init__(self, dataset_path: str, pred_horizon: int, obs_horizon: int):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.sequence_len = obs_horizon + pred_horizon 
        self.train_data = {}
        self.stats = {}
        self._create_dataset(dataset_path)

    def _create_dataset(self, dataset_path):
        train_image_data, train_data, episode_ends = self._load_data(dataset_path) 

        self.indices = create_sample_indices_sparse(
            ends=episode_ends,
            sequence_length=self.sequence_len,
            step_size=10
            )
        
        self.stats = self._compute_stats(train_data)
        normalized_action_data, normalized_velocity_data = self._normalize_data(train_data)
        
        self.train_data = {
            'position': train_data['position'],
            'velocity': normalized_velocity_data,
            'action': normalized_action_data,
            'image': train_image_data
        }

    def _load_data(self, dataset_path):
        print("*** Loading Data from {} ***".format(dataset_path))
        dataset_root = zarr.open(dataset_path, 'r')
        train_image_data = np.moveaxis(dataset_root['data']['img'][:], -1, 1)
        train_data = {
            'position': dataset_root['data']['position'][:],
            'velocity': dataset_root['data']['velocity'][:],
            'action': dataset_root['data']['action'][:],
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]
        print("*** Data Loaded ***")
        return train_image_data, train_data, episode_ends
    

    def _compute_stats(self, train_data):
        sample = None
        position_min = []
        position_max = []
        for i in range(len(self.indices)):
            start_idx, end_idx, _, _ = self.indices[i]
            sample = sample_sequence_array_sparse(
                data_array=train_data['position'],
                sample_start_idx= start_idx,
                sample_end_idx= end_idx,
                step_size=10,
            )
            local_stats = get_data_stats(sample)
            position_max.append(local_stats['max'])
            position_min.append(local_stats['min'])
        pos_stats = {'max': np.average(position_max), 'min': np.average(position_min)}
        action_stats = get_data_stats(train_data['action'])
        vel_stats = get_data_stats(train_data['velocity'])
        stats = {'position': pos_stats, 'velocity': vel_stats, 'action': action_stats}
    
        return stats

    def _normalize_data(self, train_data):
        normalized_action_data = normalize_data(train_data['action'], stats=self.stats['action'])
        normalized_velocity_data = normalize_data(train_data['velocity'], stats=self.stats['velocity'])
        return normalized_action_data, normalized_velocity_data
        
    def _normalize_sample(self, nsample):
        sample_normalized = normalize_data(nsample['position'], self.stats['position'])
        translation_vec = sample_normalized[0, :]
        nsample['position'] = (sample_normalized - translation_vec) / 2.0
        return nsample

    def __getitem__(self, idx):
        start_idx, end_idx, _ , _ = self.indices[idx]
        nsample = sample_sequence_sparse(
                data=self.train_data,
                sample_start_idx= start_idx,
                sample_end_idx= end_idx,
                step_size=10,
            )
        return self._normalize_sample(nsample)

    def __len__(self):
        return len(self.indices)

class CarRacingDatasetForInference(CarRacingDataset):
    def __init__(self, dataset_path: str, pred_horizon: int, obs_horizon: int, stats: dict):
        super().__init__(dataset_path, pred_horizon, obs_horizon)
        self.stats = stats
        self._create_dataset(dataset_path)

    def _create_dataset(self, dataset_path):
        train_image_data, train_data, episode_ends = self._load_data(dataset_path) 

        self.indices = create_sample_indices_sparse(
            ends=episode_ends,
            sequence_length=self.sequence_len,
            step_size=10
        )
        
        normalized_action_data, normalized_velocity_data = self._normalize_data(train_data)
        
        self.train_data = {
            'position': train_data['position'],
            'velocity': normalized_velocity_data,
            'action': normalized_action_data,
            'image': train_image_data
        }

    def _normalize_sample(self, nsample):
        sample_normalized = normalize_data(nsample['position'], self.stats['position'])
        translation_vec = sample_normalized[0, :]
        nsample_centered = sample_normalized - translation_vec
        nsample['position'] = nsample_centered / 2.0
        return nsample, translation_vec

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, _, _ = self.indices[idx]

        nsample = sample_sequence_sparse(
            data=self.train_data,
            sample_start_idx=buffer_start_idx,
            sample_end_idx=buffer_end_idx,
            step_size=10,
        )

        sample , translation = self._normalize_sample(nsample)
        return sample , translation

class CarRacingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str = "path/to/dir", T_obs=4, T_pred=8, seed=None, stats=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.seed = seed
        self.stats = stats

    def setup(self, name: str = None):
        dataset_path = os.path.join(self.data_dir, name)

        if self.stats:
            self.data_full = CarRacingDatasetForInference(dataset_path, self.T_pred, self.T_obs, self.stats)
        else: 
            self.data_full = CarRacingDataset(dataset_path, self.T_pred, self.T_obs)
            self.stats = self.data_full.stats

        train_length = int(len(self.data_full) * 0.8)
        val_length = len(self.data_full) - train_length
        generator = torch.Generator().manual_seed(self.seed) if self.seed else None

        self.data_train, self.data_val = random_split(self.data_full, [train_length, val_length], generator=generator)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def save_stats(self, path):
        with open(path, 'wb') as f:
            pickle.dump([self.stats], f)


            














