import numpy as np
import pickle
import zarr
import torch

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