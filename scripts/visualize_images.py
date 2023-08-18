import zarr
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from PIL import Image

dataset_path = "./data/multipleDrivingBehaviours_testing_20eps_normalized.zarr.zip"
# read from zarr dataset
dataset_root = zarr.open(dataset_path, 'r')

# float32, [0,1], (N,96,96,3)
train_image_data = dataset_root['data']['img'][:]
#train_image_data = np.moveaxis(train_image_data, -1,1)
# (N,3,96,96) Meaning N images, 3 channels, 96x96 pixels
#train_actions_data = dataset_root['data']['action'][:] # (N,3)

# Plot images:
#plt.imshow(train_image_data[100])


image_size = 80
#Crop image out of the center
cropped_img = train_image_data[100][:80, :80, :]
plt.imshow(cropped_img)
plt.show()
plt.waitforbuttonpress()


# reverse_transform = Compose([
#                     Lambda(lambda t: (t + 1) / 2),
#                     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#                     Lambda(lambda t: t * 255.),
#                     Lambda(lambda t: t.numpy().astype(np.uint8)),
#                     ToPILImage(),
#                     ])