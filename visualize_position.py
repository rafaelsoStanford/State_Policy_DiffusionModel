import zarr
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np

def normalize_batch(batch):
    # Calculate minimum and maximum values for each vector in the batch
    mins = np.min(batch, axis=(1, 2), keepdims=True)
    maxs = np.max(batch, axis=(1, 2), keepdims=True)
    
    # Normalize each vector in the batch
    normalized_batch = (batch - mins) / (maxs - mins)
    
    return normalized_batch


def sample_sequence(train_data, sequence_length,
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


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    # ndata = ndata * 2 - 1
    return ndata


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

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


dataset_path = "./data/ThreeBehaviours_20Eps.zarr.zip"
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


# Visualize the data

pos_track_1 = train_data['position'][0:episode_ends[0]]
pos_track_2 = train_data['position'][episode_ends[0]:episode_ends[1]]
pos_track_3 = train_data['position'][episode_ends[1]:episode_ends[2]]
pos_track_4 = train_data['position'][episode_ends[2]:episode_ends[3]]


plt.plot(pos_track_1[:,0], pos_track_1[:,1])
plt.plot(pos_track_2[:,0], pos_track_2[:,1], 'r')
plt.plot(pos_track_3[:,0], pos_track_3[:,1], 'g')
plt.plot(pos_track_4[:,0], pos_track_4[:,1], 'y')
plt.legend(['Track 1', 'Track 2', 'Track 3', 'Track 4'])
plt.waitforbuttonpress()
plt.close()


# # Normalized data to [-1,1], images are assumed to be normalized
# indices = create_sample_indices(
#             episode_ends=episode_ends,
#             sequence_length= 100,
#             pad_before= 0,
#             pad_after= 0)

# # ========== Normalize Data ============ 
# # normalized data to [-1,1], images are assumed to be normalized 
# stats = dict()
# normalized_train_data = dict()
# for key, data in train_data.items():
stats = get_data_stats(train_data['position'])
normalized_train_data = normalize_data(train_data['position'], stats)

# ========== Visualize Data ============
plt.plot(normalized_train_data[0:episode_ends[0],0], normalized_train_data[0:episode_ends[0],1])
plt.plot(normalized_train_data[episode_ends[0]:episode_ends[1],0], normalized_train_data[episode_ends[0]:episode_ends[1],1], 'r')
plt.plot(normalized_train_data[episode_ends[1]:episode_ends[2],0], normalized_train_data[episode_ends[1]:episode_ends[2],1], 'g')
plt.plot(normalized_train_data[episode_ends[2]:episode_ends[3],0], normalized_train_data[episode_ends[2]:episode_ends[3],1], 'y')
plt.legend(['Track 1', 'Track 2', 'Track 3', 'Track 4'])
plt.waitforbuttonpress()
plt.close()


# Get Segment

# ========== Create Sample Indices ============
sequence_length = 100

indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length= sequence_length,
            pad_before= 0,
            pad_after= 0)

np.random.shuffle(indices) # shuffle indices
stat_hist_min = []
stat_hist_max = []
for i in range(100):            
    # get the start/end indices for this datapoint
    buffer_start_idx, buffer_end_idx, \
        sample_start_idx, sample_end_idx = indices[i]

    sample = sample_sequence(
                train_data= train_data['position'],
                sequence_length=    sequence_length,
                buffer_start_idx=   buffer_start_idx,
                buffer_end_idx=     buffer_end_idx,
                sample_start_idx=   sample_start_idx,
                sample_end_idx=     sample_end_idx
            )

    # ========== normalize sample ============

    sample_stat = get_data_stats(sample)
    min = sample_stat['min']
    max = sample_stat['max']
    stat_hist_min.append(min)
    stat_hist_max.append(max)

    sample_normalized = normalize_data(sample, sample_stat)
    translation_vec = sample_normalized[0]
    sample_centered = sample_normalized - translation_vec

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(sample[:,0], sample[:,1])
    ax2.plot(sample_normalized[:,0], sample_normalized[:,1])
    ax3.plot(sample_centered[:,0], sample_centered[:,1])
    # Point at origin
    ax3.scatter(0,0, c='r')
    
    #plt.waitforbuttonpress()
    plt.close()

print(np.mean(stat_hist_max))
print(np.mean(stat_hist_min))
# ========== Create Sample Batches ============
# batch_size = 10
# # create batches
# batch = torch.zeros((batch_size, sequence_length, 2))

# for i in range(batch_size):
#     # get the start/end indices for this datapoint
#     buffer_start_idx, buffer_end_idx, \
#         sample_start_idx, sample_end_idx = indices[i]

#     batch[i,...] = torch.tensor( sample_sequence(
#                 train_data= train_data['position'],
#                 sequence_length=    sequence_length,
#                 buffer_start_idx=   buffer_start_idx,
#                 buffer_end_idx=     buffer_end_idx,
#                 sample_start_idx=   sample_start_idx,
#                 sample_end_idx=     sample_end_idx
#             ))  

# print(batch.shape)

# for traj in range(batch.shape[0]):

#     # ========== normalize sample ============
#     sample_stat = get_data_stats(batch[traj, ...])
#     sample_normalized = normalize_data(batch[traj, ...], sample_stat)
#     translation_vec = sample_normalized[0]
#     sample_centered = sample_normalized - translation_vec

#     plt.plot(sample_centered[: , 0], sample_centered[: , 1])
#     plt.waitforbuttonpress()
#     plt.close()

for i in range(2):            
    # get the start/end indices for this datapoint
    buffer_start_idx, buffer_end_idx, \
        sample_start_idx, sample_end_idx = indices[i]

    sample = sample_sequence(
                train_data= train_data['position'],
                sequence_length=    sequence_length,
                buffer_start_idx=   buffer_start_idx,
                buffer_end_idx=     buffer_end_idx,
                sample_start_idx=   sample_start_idx,
                sample_end_idx=     sample_end_idx
            )

    # ========== normalize sample ============

    sample_stat = {
        'min': -17.8,
        'max': 39.4
    }


    sample_normalized = normalize_data(sample, sample_stat)
    translation_vec = sample_normalized[0]
    sample_centered = sample_normalized - translation_vec

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(sample[:,0], sample[:,1])
    ax2.plot(sample_normalized[:,0], sample_normalized[:,1])
    ax3.plot(sample_centered[:,0], sample_centered[:,1])
    # Point at origin
    ax3.scatter(0,0, c='r')
    
    plt.waitforbuttonpress()
    plt.close()