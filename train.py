import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models.diffusion_ddpm import *
from utils.load_data import *
from utils.print_utils import *

# =========== parser function ===========
def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    parser.add_argument('--obs_horizon', type=int, default=10, help='Observation horizon')
    parser.add_argument('--pred_horizon', type=int, default=30, help='Prediction horizon')
    parser.add_argument('--action_horizon', type=int, default=1, help='Action horizon')
    parser.add_argument('--inpaint_horizon', type=int, default= 1, help='Inpaining horizon, which denotes the amount of steps of our observations to use for inpainting')
    parser.add_argument('--step_size', type=int, default=5, help='Rate of sampling from the dataset') # 10 equal to 0.2s
    parser.add_argument('--noise_steps', type=int, default=1000, help='Denoising steps')
    
    parser.add_argument('--cond_dim', type=int, default=128+2+3+2, help='Dimension of diffusion input state')
    parser.add_argument('--output_dim', type=int, default=5, help='Dimension of diffusion output state')
    parser.add_argument('--model', type=str, default='UNet_Film', help='String for choosing model architecture')
    parser.add_argument('--noise_scheduler', type=str, default='linear', help='String for choosing noise scheduler')

    parser.add_argument('--dataset_dir', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--dataset', type=str, default='2023-07-30-1836_dataset_20_episodes_2_modes.zarr.zip', help='zarr.zip dataset filename')
    
    return parser.parse_args()

############################
#========== MAIN ===========
############################
def main(args):
    # ===========Parameters===========
    # training parameters
    n_epochs = args.n_epochs
    AMP = args.amp
    batch_size = args.batch_size
    lr = args.lr

    # diffusion parameters
    noise_steps = args.noise_steps
    obs_horizon = args.obs_horizon
    pred_horizon = args.pred_horizon
    inpaint_horizon = args.inpaint_horizon
    action_horizon = args.action_horizon
    cond_dim = args.cond_dim
    output_dim = args.output_dim
    step_size = args.step_size

    # Dataset dir and filename
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset
    tb_dir = "tb_logs/"

    # model architecture
    model = args.model
    noise_scheduler = args.noise_scheduler
    
    # =========== Loading Data ===========
    # Load Dataset using Pytorch Lightning DataModule
    dataset = CarRacingDataModule(batch_size, dataset_dir , obs_horizon, pred_horizon ,action_horizon, step_size=step_size)
    
    dataset.setup(name=dataset_name)
    train_dataloader = dataset.train_dataloader()
    valid_dataloader = dataset.val_dataloader()


    # # ===========model===========
    diffusion = Diffusion_DDPM(
                    noise_steps= noise_steps,
                    obs_horizon=obs_horizon,
                    pred_horizon= pred_horizon,
                    observation_dim=cond_dim,
                    prediction_dim= output_dim,
                    model=model,
                    learning_rate=lr,
                    inpaint_horizon=inpaint_horizon,
                    noise_scheduler=noise_scheduler,
                    step_size=step_size,
    )
    
    # ===========trainer===========
    # -----PL configs-----
    tensorboard = pl_loggers.TensorBoardLogger(save_dir=tb_dir ,name='',flush_secs=1)
    early_stop_callback = EarlyStopping(monitor='lr', stopping_threshold=1e-4, patience=n_epochs//10, verbose=True)   
    checkpoint_callback = ModelCheckpoint(filename="{epoch}",     # Checkpoint filename format
                                          save_top_k=-1,          # Save all checkpoints
                                          every_n_epochs=1,               # Save every epoch
                                          save_on_train_epoch_end=True,
                                          verbose=True)

    
    # -----train model-----
    trainer = pl.Trainer(accelerator='gpu', devices=[0], precision=("16-mixed" if AMP else 32), max_epochs=n_epochs, 
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=tensorboard, profiler="simple", val_check_interval=0.25, 
                         accumulate_grad_batches=1, gradient_clip_val=0.5) 

    FPS = 50
    print(" Further Info: Observation Horizon: {}, Prediction Horizon: {}, Inpaint Horizon: {}".format(obs_horizon, pred_horizon, inpaint_horizon))
    print(" Step size: {}".format(dataset.data_full.step_size))
    print(" In Seconds: Observation Horizon {} s".format(dataset.data_full.step_size*obs_horizon*1/FPS))
    print(" In Seconds: Prediction Horizon {} s".format(dataset.data_full.step_size*pred_horizon*1/FPS))
    print(" In Seconds: Inpaint Horizon {} s".format(dataset.data_full.step_size*inpaint_horizon*1/FPS))
    

    # -----print model info------
    if os.getenv("LOCAL_RANK", '0') == '0':
        print_dataset_info(args,dataset_dir, dataset_name, train_dataloader, tensorboard)

    # -----save stats------
    trainer.validate(model= diffusion, dataloaders=valid_dataloader)
    
    tb_new_dir = tensorboard.log_dir  #.validate() had to be called before, otherwise tb_new_dir is not yet created
    dataset.save_stats( tb_new_dir + "/STATS.pkl")
    
    # -----train model-----
    trainer.fit(model=diffusion, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)





