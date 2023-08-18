import os
from models.diffusion_ddpm import *
from models.diffusion_ddim import *
from utils.load_data import *
import pickle
import time 
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DDIM', help="Can be 'DDPM', 'DDIM'")
    parser.add_argument('--version', type=str, default='860', help='Version control variable')
    parser.add_argument('--checkpoint_epoch', type=str, default='4', help='Checkpoint epoch')
    parser.add_argument('--stats_file_name', type=str, default='STATS.pkl', help='Stats file name')
    parser.add_argument('--dataset_dir', type=str, default='./data', help='Directory of the dataset')
    parser.add_argument('--dataset_name', type=str, default='2023-07-17-2252_dataset_1_episodes_2_modes.zarr.zip', help='Name of the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--seed', type=int, default=125, help='Random seed')
    args = parser.parse_args()
    return args

def load_model(model_name, checkpoint_path, hparams_path, num_of_ddim_steps = 100):
    if model_name == 'DDPM':
        model = Diffusion_DDPM.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path)
    if model_name == 'DDIM':
        model = Diffusion_DDIM.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path)
        noise_scheduler = DDIMScheduler(
                num_train_timesteps= num_of_ddim_steps, # 1000
                beta_schedule= 'linear',#'squaredcos_cap_v2', # 'cosine_beta_schedule'
                clip_sample=False, # clip to [-1, 1]
                prediction_type='epsilon', # 'predicting error'
            )
        model.noise_scheduler = noise_scheduler # overwrite the ddpm noise scheduler with ddim noise scheduler
        model.noise_steps = num_of_ddim_steps # Slight abuse of the use of noise steps; Noise steps was used for DDPM training, now storing the desired number of DDIM steps
    model.eval()
    return model


# ==========================================
# ================== Main ==================
# ==========================================

args = parse_arguments()

BASE_PATH           = './tb_logs/version_' + args.version
HPARAMS_PATH        = os.path.join(BASE_PATH, 'hparams.yaml')
CHECKPOINT_PATH     = os.path.join(BASE_PATH, 'checkpoints', f'epoch={args.checkpoint_epoch}.ckpt')
STATS_PATH          = os.path.join(BASE_PATH, args.stats_file_name)
SAVING_PATH         = './animations'

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
# During inference, Model input only needs to be observation horizon
observation_batch = model.prepare_observation_batch(batch[0])

# ----------------- Sampling -----------------
print(f'***Sampling with {args.model_name}...')
start = time.time()
if args.model_name == 'DDIM':
    sampling_history = model.sample(batch=observation_batch, option='sample_history')
if args.model_name == 'DDPM':
    sampling_history = model.sample(batch=observation_batch, option='sample_history')

end = time.time()
print(f'*** Time taken for sampling: {end-start} ***')

save_dir = os.path.join(SAVING_PATH, args.model_name)
os.makedirs(save_dir, exist_ok=True)
plt_toVideo(model, sampling_history, batch, save_dir)

