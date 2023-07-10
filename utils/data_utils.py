
import torch
import numpy as np

def get_data_stats(data: torch.Tensor):
    data = data.reshape(-1,data.shape[-1])
    min, _ =  torch.min(data, axis=0)
    max, _ =  torch.max(data, axis=0)
    stats = {
        'min': min,
        'max': max
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    diffs = stats['max'] - stats['min']
    if torch.any(diffs == 0): # Avoid division by zero
        diffs[diffs == 0] = 1
    ndata = (data - stats['min']) / diffs
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def normalize_batch(data, stats):
    # stats has B elements
    ndata = torch.zeros_like(data)
    for b, d in enumerate(data):
        ndata[b,...] = normalize_data(d, stats[b])
    return ndata

def normalize_image(image):
    """
    Normalize image data assuming values are already between 0 and 1.
    """
    assert (image.max() <= 1.0 and image.min() >= 0.0)
    return image

def normalize_position(position: torch.Tensor):
    """
    Normalize position data by subtracting the mean and dividing by 2.
    position: (B, T, 2)
    """
    stats_list = []
    position_normalized = torch.zeros_like(position)
    translation_to_zero = torch.zeros((position.size(0), 2), device=position.device)
    for b, data in enumerate(position):
        stats = get_data_stats(data)  # Assuming get_data_stats function is defined
        sample_normalized = normalize_data(data, stats)
        translation_to_zero[b] = sample_normalized[0,:]
        sample_normalized = (sample_normalized - translation_to_zero[b]) / 2.0
        stats_list.append(stats)
        position_normalized[b] = sample_normalized
    assert(not torch.isnan(position_normalized).any())
    return position_normalized, translation_to_zero, stats_list

def normalize_action(action):
    # Normalize action data assuming values are already between -1 and 1, when loaded by the dataloader.
    assert (action.max() <= 1.0 and action.min() >= -1.0)
    assert(not torch.isnan(action).any())
    return action


def normalize_velocity(velocity):
    # Normalize velocity data assuming values are already between -1 and 1, done by the dataloader.
    assert (velocity.max() <= 1.0 and velocity.min() >= -1.0)
    assert(not torch.isnan(velocity).any())
    return velocity



