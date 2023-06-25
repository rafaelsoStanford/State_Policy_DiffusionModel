import zarr
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np


dataset_path = "./data/twoBehaviours_testing_5eps_normalized.zarr.zip"
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


# Visualize the action data
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(train_data['action'][:,0])
ax1.scatter( np.arange(train_data['action'].shape[0]), train_data['action'][:,0] , c='r', s=1)
ax2.plot(train_data['action'][:,1])
ax2.scatter( np.arange(train_data['action'].shape[0]), train_data['action'][:,1] , c='r', s=1)
ax3.plot(train_data['action'][:,2])
ax3.scatter( np.arange(train_data['action'].shape[0]), train_data['action'][:,2] , c='r', s=1)

plt.waitforbuttonpress()
plt.close()