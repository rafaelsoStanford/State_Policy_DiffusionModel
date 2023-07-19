import numpy as np
import cv2
import yaml
import zarr
import shutil
import os

def maskTrajectories(image):
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


def fetch_hyperparams_from_yaml(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams

def calculate_target_point(augmImg, strip_distance, car_pos_vector, MODE):
    dict_masks = maskTrajectories(augmImg) # Get dictionary of masks containing each trajectory
    track_img = dict_masks[MODE] # Get the mask of the trajectory we are following
    line_strip = track_img[strip_distance, :]
    idx = np.nonzero(line_strip)[0]
    
    if len(idx) == 0:
        return None
    
    idx = idx[np.argmin(np.abs(idx - 48))]
    target_point = np.array([strip_distance, idx])
    return target_point

def calculate_car2point_vector(target_point, car_pos_vector):
    return target_point - car_pos_vector

def calculate_steering_angle(error_avg_2, car2point_vector):
    angle = np.arctan2(abs(error_avg_2), abs(car2point_vector[0]))
    if error_avg_2 > 0:
        angle = -angle
    return angle

def calculate_steering_action(angle, pid_steering):
    return pid_steering(angle)

def calculate_velocity_error(pid_velocity, v_wFrame):
    return pid_velocity.setpoint - np.linalg.norm(v_wFrame)

def calculate_velocity_action(error_vel_avg, pid_velocity, v_wFrame):
    if error_vel_avg < 0:
        return 0, np.clip(np.linalg.norm(pid_velocity(np.linalg.norm(v_wFrame))), 0, 0.9)
    else:
        return pid_velocity(np.linalg.norm(v_wFrame)), 0

def trajectory_control(augmImg, pid_steering, pid_velocity, 
                       error_buffer, error_buffer_2, error_velocity_buffer, v_wFrame, MODE):
    
    strip_distance = 60 # x - coordinates of the strip, ie image row starting in upper left corner
    car_pos_vector = np.array([70, 48])
    
    target_point = calculate_target_point(augmImg, strip_distance, car_pos_vector, MODE)
    if target_point is None:
        return None
    
    car2point_vector = calculate_car2point_vector(target_point, car_pos_vector)
    err = target_point[1] - 48.0
    err = np.clip(err, -5, 5)
    
    if np.linalg.norm(err) <= 2:
        err = 0.3 * err
    
    error_buffer.append(err)
    error_avg = sum(error_buffer) / len(error_buffer)
    error_buffer_2.append(error_avg)
    error_avg_2 = sum(error_buffer_2) / len(error_buffer_2)
    
    angle = calculate_steering_angle(error_avg_2, car2point_vector)
    action_steering = calculate_steering_action(angle, pid_steering)
    
    error_vel = calculate_velocity_error(pid_velocity, v_wFrame)
    if np.linalg.norm(error_vel) < 2.0:
        error_vel = 0.0 * error_vel #Attenuate the error if it is too small
    
    error_velocity_buffer.append(error_vel)
    error_vel_avg = sum(error_velocity_buffer) / len(error_velocity_buffer)
    action_velocity, action_acceleration = calculate_velocity_action(error_vel_avg, pid_velocity, v_wFrame)
    
    return [action_steering, action_velocity, action_acceleration]

def create_folder(base_dir, folder_name):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists.")

def generate_dataset_name(dataset_name, folder_name, NUM_EPISODES_PER_MODE, num_modes):
    if dataset_name:
        # Stitch dataset name with additional information
        dataset_name = dataset_name.rstrip('.zarr')
        dataset_name = dataset_name + f"_dataset_{NUM_EPISODES_PER_MODE}_episodes_{num_modes}_modes.zarr"
    else:
        # If no dataset name is provided, create a new one
        dataset_name = folder_name + f"_dataset_{NUM_EPISODES_PER_MODE}_episodes_{num_modes}_modes.zarr"
    return dataset_name

def save_buffer_to_zarr(buffer, base_dir, folder_name, dataset_name, chunk_len):
    path = os.path.join(base_dir, folder_name, dataset_name)
    print("Saving data to path:", path)
    buffer.save_to_path(path, chunk_length=chunk_len)

    # Consolidate metadata
    store = zarr.DirectoryStore(path)
    zarr.consolidate_metadata(store)

    # Zip the file
    zarr_file = os.path.basename(path)
    zip_file = zarr_file + ".zip"
    shutil.make_archive(path, "zip", path)
    print(f"Zarr file saved as {zip_file}")

def extract_variables(info):
    augmImg = info['augmented_img']
    velB2vec = info['car_velocity_vector']
    posB2vec = info['car_position_vector']
    car_heading_angle = info['car_init_angle']

    carVelocity_wFrame = [velB2vec.x, velB2vec.y]
    carPosition_wFrame = [posB2vec.x, posB2vec.y]
    v_wFrame = np.linalg.norm(velB2vec)

    return augmImg, carVelocity_wFrame, carPosition_wFrame, car_heading_angle, v_wFrame

def append_to_histories(observation, velocity, position, action, angle, img_hist, vel_hist, pos_hist, act_hist, angle_hist):
    img_hist.append(observation)
    vel_hist.append(velocity)
    pos_hist.append(position)
    act_hist.append(action.copy())
    angle_hist.append(angle)

def save_data_to_buffer(buffer, img_hist, act_hist, vel_hist, pos_hist, angle_hist):
    img_hist = np.array(img_hist, dtype=np.float32)
    act_hist = np.array(act_hist, dtype=np.float32)
    vel_hist = np.array(vel_hist, dtype=np.float32)
    pos_hist = np.array(pos_hist, dtype=np.float32)
    angle_hist = np.array(angle_hist, dtype=np.float32)

    img_hist = img_hist / 255.0  # Normalize images (expected by model)

    episode_data = {
        "img": img_hist,
        "velocity": vel_hist,
        "position": pos_hist,
        "action": act_hist,
        "angle": angle_hist,
    }

    buffer.add_episode(episode_data)
    print("Episode finished after {} timesteps".format(len(img_hist)))