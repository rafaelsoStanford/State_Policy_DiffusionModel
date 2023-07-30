import os
from models.diffusion_ddpm import *
from models.diffusion_ddim import *
from utils.load_data import *
import pickle
import time 
import argparse


def load_model(model_name, checkpoint_path, hparams_path):
    if model_name == 'DDPM':
        model = Diffusion_DDPM.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path)
    elif model_name == 'DDIM' or model_name == 'DDIPM':
        model = Diffusion_DDIM.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path)
    else:
        raise ValueError("Model name must be either 'DDPM', 'DDIM', or 'DDIPM'")
    model.eval()
    return model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DDPM', help="Can be 'DDPM', 'DDIM', or 'DDIPM'")
    parser.add_argument('--version', type=str, default='659', help='Version control variable')
    parser.add_argument('--checkpoint_epoch', type=str, default='12', help='Checkpoint epoch')
    parser.add_argument('--stats_file_name', type=str, default='STATS.pkl', help='Stats file name')
    parser.add_argument('--dataset_dir', type=str, default='./data', help='Directory of the dataset')
    parser.add_argument('--dataset_name', type=str, default='2023-07-17-2252_dataset_1_episodes_2_modes.zarr.zip', help='Name of the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--seed', type=int, default=125, help='Random seed')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    BASE_PATH = './tb_logs/version_' + args.version
    HPARAMS_PATH = os.path.join(BASE_PATH, 'hparams.yaml')
    CHECKPOINT_PATH = os.path.join(BASE_PATH, 'checkpoints', f'epoch={args.checkpoint_epoch}.ckpt')
    STATS_PATH = os.path.join(BASE_PATH, args.stats_file_name)
    SAVING_PATH = './animations'

    with open(STATS_PATH, 'rb') as f:
        stats = pickle.load(f)
    stats = stats[0]

    model = load_model(args.model_name, CHECKPOINT_PATH, HPARAMS_PATH)

    model_params = fetch_hyperparams_from_yaml(HPARAMS_PATH)
    obs_horizon = model_params['obs_horizon']
    pred_horizon = model_params['pred_horizon']

    dataset = CarRacingDataModule(args.batch_size, args.dataset_dir, obs_horizon, pred_horizon, seed=args.seed, stats=stats)
    dataset.setup(name=args.dataset_name)
    test_dataloaders = dataset.val_dataloader()

    batch = next(iter(test_dataloaders))

    print(f'***Sampling with {args.model_name}...')
    start = time.time()
    if args.model_name == 'DDPM':
        sampling_history = model.sample(batch=batch[0], mode='sampling')
    elif args.model_name == 'DDIM':
        sampling_history = model.sample(batch=batch[0], mode='sampling', step_size=50)
    elif args.model_name == 'DDIPM':
        sampling_history = model.sample(batch=batch[0], mode='sampling', step_size=50, ddpm_steps=100)
    end = time.time()
    print("     ... Sampling complete! ***")
    print(f'*** Time taken for sampling: {end-start} ***')

    save_dir = os.path.join(SAVING_PATH, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    plt_toVideo(model, sampling_history, batch, save_dir)


if __name__ == "__main__":
    main()

