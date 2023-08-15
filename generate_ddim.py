from models.diffusion_ddpm import *
from models.diffusion_ddim import *
from utils.load_data import *




def plt_toVideo(self, 
                sampling_history,
                batch,
                saving_path,):
    
    position_observation    = batch[0]['position'].squeeze()[:self.obs_horizon].detach().cpu().numpy() #(20 , 2)
    positions_groundtruth   = batch[0]['position'].squeeze().detach().cpu().numpy() #(20 , 2)
    actions_groundtruth     = batch[0]['action'].squeeze().detach().cpu().numpy() #(20 , 3)
    actions_observation     = batch[0]['action'].squeeze()[:self.obs_horizon].detach().cpu().numpy() #(20 , 3)
    # ---------------- Plotting ----------------
    # 
    sampling_positions = np.array(sampling_history)[:, :, :2]  # (1000, 45 , 2)
    sampling_actions = np.array(sampling_history)[:, :, 2:]  # (1000, 45 , 3)

    def plot_positions():
        fig, ax = plt.subplots()

        cmap = plt.get_cmap('viridis', self.pred_horizon + self.inpaint_horizon)
        indices = np.arange(self.pred_horizon + self.inpaint_horizon)

        def animate(frame):
            fig.clf()
            normalized_indices = indices / (self.pred_horizon + self.inpaint_horizon - 1)
            colors = cmap(normalized_indices)


            plt.plot(positions_groundtruth[:, 0], positions_groundtruth[:, 1], 'g.')
            plt.plot(position_observation[:, 0], position_observation[:, 1], 'b.')
            plt.scatter(sampling_positions[frame, :, 0], sampling_positions[frame, :, 1], color=colors, s=20)

            plt.grid()
            plt.axis('equal')
            
            # Use ground truth to define axis limits
            plt.xlim(positions_groundtruth[:, 0].min(), positions_groundtruth[:, 0].max())
            plt.ylim(positions_groundtruth[:, 1].min(), positions_groundtruth[:, 1].max())

        fig.animation = FuncAnimation(fig, animate, frames=len(sampling_history), interval=20, repeat=False, blit=False)
        # fig.animation.save('./animations/' + self.date + 'animation_positions.gif', writer='pillow')
        # print("Animation saved")
        # plt.close('all')

        # Save the animation as an MP4 file
        fig.animation.save(saving_path + '/' + self.date + 'animation_positions.mp4', writer='ffmpeg')

        print("Positions animation saved")



    def plot_actions():
        fig2, ax1 = plt.subplots()

        def animate_actions(frame):
            fig2.clf()

            plt.plot(actions_groundtruth[:, 0])
            plt.plot(actions_observation[:, 0])
            # ax2.plot(actions_groundtruth[ :, 1])
            # ax3.plot(actions_groundtruth[ :, 2])
            # inpaint_start = 0
            # inpaint_end = self.inpaint_horizon
            # plt.axvspan(inpaint_start, inpaint_end, alpha=0.2, color='red')
            # plt.axvspan(inpaint_end, sampling_actions.shape[1], alpha=0.2, color='green')
            #ax1.plot(sampling_actions[frame, :, 0])
            plt.scatter(np.arange(sampling_actions.shape[1]) + (self.obs_horizon - self.inpaint_horizon), sampling_actions[frame,:,0] , c='g', s=10)

            # ax3.plot(sampling_actions[frame, :, 2])
            plt.grid()
            plt.ylim(-1.5, 1.5)

        fig2.animation = FuncAnimation(fig2, animate_actions, frames=len(sampling_history), interval=20, repeat=False, blit=False)
        
        fig2.animation.save(saving_path + '/' + self.date + 'animation_actions.mp4', writer='ffmpeg')

        print("Actions animation saved")
        plt.close('all')

    plot_actions()
    plot_positions()

############################
#========== MAIN ===========
############################

def main():
    batch_size = 1

    # =========== Load Model ===========
    # ? path_hyperparams = './tb_logs/version_588/hparams.yaml'
    # ? path_checkpoint = './tb_logs/version_588/checkpoints/epoch=55.ckpt'
    
    path_hyperparams = './tb_logs/version_651/hparams.yaml'
    path_checkpoint = './tb_logs/version_651/checkpoints/epoch=17.ckpt'
    
    dataset_name = '2023-07-18-0031_dataset_1_episodes_2_modes.zarr.zip'

    model = Diffusion_DDIM.load_from_checkpoint( #Choose between DDPM and DDIM -- Model is inherited from DDPM thus they should be compatible
        path_checkpoint,
        hparams_file=path_hyperparams,
    )
    model.eval() 

    # ===========Parameters===========
    model_params = fetch_hyperparams_from_yaml(path_hyperparams)
    obs_horizon = model_params['obs_horizon']
    pred_horizon = model_params['pred_horizon']

    # =========== Dataloader ===========
    # Dataset dir and filename
    dataset_dir = './data'
    dataset = CarRacingDataModule( batch_size, dataset_dir, obs_horizon, pred_horizon ,seed=125)
    dataset.setup( name = dataset_name )
    test_dataloaders = dataset.val_dataloader()

    batch = next(iter(test_dataloaders))
    sampling_history = model.sample(batch=batch, step_size=50, ddpm_steps = 100)



if __name__ == "__main__":
    main()
    