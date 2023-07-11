import cv2
import numpy as np


def maskTrajecories(image):
    # Define the threshold ranges for each mask
    lower_yellow = np.array([100, 100, 0], dtype=np.uint8) ## Left of track in grass
    upper_yellow = np.array([255, 255, 0], dtype=np.uint8)
    lower_cyan = np.array([0, 100, 100], dtype=np.uint8) ## Left side of track on road
    upper_cyan = np.array([0, 255, 255], dtype=np.uint8)
    lower_magenta = np.array([100, 0, 100], dtype=np.uint8) ## Middle of track on road
    upper_magenta = np.array([255, 0, 255], dtype=np.uint8)
    lower_purple = np.array([100, 0 , 100], dtype=np.uint8) ## Right side of track on road
    upper_purple = np.array([200, 50,200], dtype=np.uint8)
    lower_blue = np.array([0, 0, 100], dtype=np.uint8) ## Right of track in grass
    upper_blue = np.array([0, 0, 255], dtype=np.uint8)

    # Apply the masks
    mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(image, lower_blue, upper_blue)
    mask_cyan = cv2.inRange(image, lower_cyan, upper_cyan)
    mask_magenta = cv2.inRange(image, lower_magenta, upper_magenta)
    mask_purple = cv2.inRange(image, lower_purple, upper_purple)

    # Label the differently colored trajectories
    dict_masks = {'lleft': mask_yellow, 
                  'left': mask_cyan, 
                  'middle': mask_magenta, 
                  'right': mask_purple, 
                  'rright': mask_blue}
    return dict_masks
