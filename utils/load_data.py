import numpy as np
import os
import pickle
import zarr
import torch
import yaml

from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

def fetch_hyperparams_from_yaml(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams

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

def sample_sequence_array(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    input_array = train_data
    sample = input_array[buffer_start_idx:buffer_end_idx]
    data = sample
    if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
        data = np.zeros(
            shape=(sequence_length,) + input_array.shape[1:],
            dtype=input_array.dtype)
        if sample_start_idx > 0:
            data[:sample_start_idx] = sample[0]
        if sample_end_idx < sequence_length:
            data[sample_end_idx:] = sample[-1]
        data[sample_start_idx:sample_end_idx] = sample
    result = data
    return result

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



# ================== Dataset loader for Traing and Validation ==================
# Path: utils/load_data.py
# Loading and normalizing data --> create dataset --> create dataloader --> save stats
class CarRacingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int
                 ):
        
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.sequence_len = obs_horizon + pred_horizon # chunk length of data
        self.train_data = {}
        self.stats = {}
        
        self._create_dataset(dataset_path)

    def _create_dataset(self, dataset_path):
        train_image_data, train_data, episode_ends = self._load_data(dataset_path) 

        self.indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=self.sequence_len,
            pad_before=0,
            pad_after=0
        )
        
        self.stats = self._compute_stats(train_data, self.indices, self.obs_horizon, self.pred_horizon)
        normalized_action_data, normalized_velocity_data = self._normalize_data(train_data)
        
        self.train_data['position'] = train_data['position']
        self.train_data['velocity'] = normalized_velocity_data
        self.train_data['action'] = normalized_action_data
        self.train_data['image'] = train_image_data
        
    def _load_data(self, dataset_path):
        print("*** Loading Data ***")
        dataset_root = zarr.open(dataset_path, 'r')
        train_image_data = np.moveaxis(dataset_root['data']['img'][:], -1, 1)
        train_data = {
            'position': dataset_root['data']['position'][:],
            'velocity': dataset_root['data']['velocity'][:],
            'action': dataset_root['data']['action'][:],
            'angle': dataset_root['data']['angle'][:],
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]

        return train_image_data, train_data, episode_ends
    
    def _compute_stats(self, train_data, indices, obs_horizon, pred_horizon):
        position_min = []
        position_max = []
        for i in range(len(indices)):
            buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[i]
            sample = sample_sequence_array(
                train_data=train_data['position'],
                sequence_length=obs_horizon + pred_horizon,
                buffer_start_idx=buffer_start_idx,
                buffer_end_idx=buffer_end_idx,
                sample_start_idx=sample_start_idx,
                sample_end_idx=sample_end_idx
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
        nsample_centered = sample_normalized - translation_vec
        nsample['position'] = nsample_centered / 2.0
        return nsample

    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.sequence_len,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
        return self._normalize_sample(nsample)

# =========== data set class for inference ===========
# During Inferrence, we load the stats from the training data set
# Small changes to the _create_dataset method
class CarRacingDataset_forInference(CarRacingDataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 stats: dict
                 ):
        
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.sequence_len = obs_horizon + pred_horizon # chunk length of data
        self.train_data = {}
        self.stats = stats
        self._create_dataset(dataset_path)
        
    def _create_dataset(self, dataset_path):
            train_image_data, train_data, episode_ends = self._load_data(dataset_path) 

            self.indices = create_sample_indices(
                episode_ends=episode_ends,
                sequence_length=self.sequence_len,
                pad_before=0,
                pad_after=0
            )
            
            normalized_action_data, normalized_velocity_data = self._normalize_data(train_data)
            
            self.train_data['position'] = train_data['position']
            self.train_data['velocity'] = normalized_velocity_data
            self.train_data['action'] = normalized_action_data
            self.train_data['image'] = train_image_data
            self.train_data['angle'] = train_data['angle']

    def _normalize_sample(self, nsample):
        sample_normalized = normalize_data(nsample['position'], self.stats['position'])
        translation_vec = sample_normalized[0, :]
        nsample_centered = sample_normalized - translation_vec
        nsample['position'] = nsample_centered / 2.0
        return nsample, translation_vec


# ====== DataModule ====== #
# ----- CarRacingDataModule is a Pytorch Lightning DataModule
class CarRacingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size,  data_dir: str = "path/to/dir" , T_obs=4, T_pred=8 , seed=None, stats=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.T_obs = T_obs
        self.T_pred = T_pred

        self.data_train = None
        self.data_val = None
        self.seed = seed
        self.stats = stats

    def setup(self, name: str = None):
        # ----- CarRacingDataset is a Dataloader file, takes care of normalization and such-----
        if self.stats:
            self.data_full = CarRacingDataset_forInference(  dataset_path= os.path.join(self.data_dir, name),
                                            pred_horizon=self.T_pred,
                                            obs_horizon=self.T_obs,
                                            stats=self.stats
                                            )
        else: 
            self.data_full = CarRacingDataset(  dataset_path= os.path.join(self.data_dir, name),
                                                pred_horizon=self.T_pred,
                                                obs_horizon=self.T_obs)
            self.stats = self.data_full.stats
            
        if self.seed:
            self.data_train, self.data_val = random_split(self.data_full, [int(len(self.data_full)*0.8), len(self.data_full) - int(len(self.data_full)*0.8)], generator=torch.Generator().manual_seed(self.seed))
        else:
            self.data_train, self.data_val = random_split(self.data_full, [int(len(self.data_full)*0.8), len(self.data_full) - int(len(self.data_full)*0.8)])
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def save_stats(self, path):
        with open(path, 'wb') as f:
            pickle.dump([self.data_full.stats], f)
            
