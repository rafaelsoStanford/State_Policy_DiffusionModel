python train.py --model="UNet_Film" --dataset="Sinusoidal_dataset_5_episodes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=20 --inpaint_horizon=5

python train.py --model="UNet_Film" --dataset="Sinusoidal_dataset_5_episodes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=20 --inpaint_horizon=10

python train.py --model="UNet_Film" --dataset="Sinusoidal_dataset_5_episodes.zarr.zip" \
 --batch_size=16 --obs_horizon=20 --pred_horizon=40 --inpaint_horizon=10

python train.py --model="UNet_Film" --dataset="TestingActions_dataset_5_episodes_3_modes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=20 --inpaint_horizon=5

python train.py --model="UNet_Film" --dataset="TestingActions_dataset_5_episodes_3_modes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=20 --inpaint_horizon=10

python train.py --model="UNet_Film" --dataset="TestingActions_dataset_5_episodes_3_modes.zarr.zip" \
 --batch_size=16 --obs_horizon=20 --pred_horizon=40 --inpaint_horizon=10


