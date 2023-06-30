import zarr
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np


dataset_path = "./data/Sinusoidal_dataset_5_episodes.zarr.zip"
# read from zarr dataset
dataset_root = zarr.open(dataset_path, 'r')

# (N, D)
train_data = {
    # Create Prediction Targets
    'position': dataset_root['data']['position'][:], # (T,2)
#            'velocities_pred': dataset_root['data']['velocity'][:] # (T,2)
    'action': dataset_root['data']['action'][:] #(T,3)
}
episode_ends = dataset_root['meta']['episode_ends'][:]

# min max
action_data = train_data['action']
minAction = np.min(action_data, axis=0)
maxAction = np.max(action_data, axis=0)

print ("minAction: ", minAction)
print ("maxAction: ", maxAction)

# Visualize the action data
fig = plt.figure()
#plt.plot(train_data['action'][0:episode_ends[0],0])
plt.scatter( np.arange(train_data['action'][0:episode_ends[0],:].shape[0]), train_data['action'][0:episode_ends[0],0] , c='r', s=1)
# ax2.plot(train_data['action'][0:episode_ends[0],1])
# ax2.scatter( np.arange(train_data['action'].shape[0]), train_data['action'][:,1] , c='r', s=1)
# ax3.plot(train_data['action'][0:episode_ends[0],2])
# ax3.scatter( np.arange(train_data['action'].shape[0]), train_data['action'][:,2] , c='r', s=1)



fig2 = plt.figure()
# Visualize the positional data
pos_track_1 = train_data['position'][0:episode_ends[0]]
#Plot track 1 in figure 2
plt.plot(pos_track_1[:,0], pos_track_1[:,1])
plt.show()
plt.waitforbuttonpress()
