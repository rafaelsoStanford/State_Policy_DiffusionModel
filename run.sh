#!/bin/bash

# Set the desired parameter values
n_epochs=500
amp=1
batch_size=16
obs_horizon=40
pred_horizon=40
action_horizon=1
cond_dim=517
output_dim=5
model="UNet"
dataset="ThreeBehaviours_20Eps.zarr.zip"

# Run the Python script with the specified arguments
python train.py \
    --n_epochs $n_epochs \
    --amp \
    --batch_size $batch_size \
    --obs_horizon $obs_horizon \
    --pred_horizon $pred_horizon \
    --action_horizon $action_horizon \
    --cond_dim $cond_dim \
    --output_dim $output_dim \
    --model $model \
    --dataset $dataset

# Set the desired parameter values
model ="UNet_Film"
# Run the Python script with the specified arguments
python train.py \
    --n_epochs $n_epochs \
    --amp \
    --batch_size $batch_size \
    --obs_horizon $obs_horizon \
    --pred_horizon $pred_horizon \
    --action_horizon $action_horizon \
    --cond_dim $cond_dim \
    --output_dim $output_dim \
    --model $model \
    --dataset $dataset
