import os
from models.diffusion_ddpm import *
from models.diffusion_ddim import *
from utils.load_data import *
import pickle
import time 

# Adjustable Variables
MODEL_NAME = 'DDPM' # Can be 'DDPM', 'DDIM', or 'DDIPM'

# Version control variables
VERSION = '659'
CHECKPOINT_EPOCH = '12'
STATS_FILE_NAME = 'STATS.pkl'
DATASET_DIR = './data'
DATASET_NAME = '2023-07-17-2252_dataset_1_episodes_2_modes.zarr.zip'

# Constants
BATCH_SIZE = 1
SEED = 125

# Paths
BASE_PATH = './tb_logs/version_' + VERSION
HPARAMS_PATH = os.path.join(BASE_PATH, 'hparams.yaml')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'checkpoints', f'epoch={CHECKPOINT_EPOCH}.ckpt')
STATS_PATH = os.path.join(BASE_PATH, STATS_FILE_NAME)
SAVING_PATH = './animations'

def load_model(model_name, checkpoint_path, hparams_path):
    if model_name == 'DDPM':
        model = Diffusion_DDPM.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path)
    elif model_name == 'DDIM' or model_name == 'DDIPM':
        model = Diffusion_DDIM.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path)
    else:
        raise ValueError("Model name must be either 'DDPM', 'DDIM', or 'DDIPM'")

    model.eval()
    return model

def main():
    # Load the pickle stats file
    with open(STATS_PATH, 'rb') as f:
        stats = pickle.load(f)
    stats = stats[0]

    model = load_model(MODEL_NAME, CHECKPOINT_PATH, HPARAMS_PATH)
    # Parameters
    model_params = fetch_hyperparams_from_yaml(HPARAMS_PATH)
    obs_horizon = model_params['obs_horizon']
    pred_horizon = model_params['pred_horizon']

    # Dataloader
    dataset = CarRacingDataModule(BATCH_SIZE, DATASET_DIR, obs_horizon, pred_horizon, seed=SEED, stats=stats)
    dataset.setup(name=DATASET_NAME)
    test_dataloaders = dataset.val_dataloader()

    batch = next(iter(test_dataloaders))

    # Sample and check the time
    print(f'***Sampling with {MODEL_NAME}...')
    start = time.time()
    if MODEL_NAME == 'DDPM':
        sampling_history = model.sample(batch=batch[0], mode='sampling')
    elif MODEL_NAME == 'DDIM':
        sampling_history = model.sample(batch=batch[0], mode='sampling', step_size=50)
    elif MODEL_NAME == 'DDIPM':
        sampling_history = model.sample(batch=batch[0], mode='sampling', step_size=50, ddpm_steps=100)
    end = time.time()
    print("     ... Sampling complete! ***")
    print(f'*** Time taken for sampling: {end-start} ***')

    # Create a new directory for saving results
    save_dir = os.path.join(SAVING_PATH, MODEL_NAME)
    os.makedirs(save_dir, exist_ok=True)

    # Save
    plt_toVideo(model, sampling_history, batch, save_dir)


if __name__ == "__main__":
    main()

