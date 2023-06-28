#!/bin/bash

# Set constant values
n_epochs=50

batch_size=16
lr=1e-4
obs_horizon=20
pred_horizon=40
action_horizon=1
inpaint_horizon=5
noise_steps=1000
cond_dim=133
output_dim=5
model="default"
dataset_dir="./data"
dataset="ThreeBehaviours_20Eps.zarr.zip"

# Run the Python script
python train.py \
    --n_epochs $n_epochs \

    --batch_size $batch_size \
    --lr $lr \
    --obs_horizon $obs_horizon \
    --pred_horizon $pred_horizon \
    --action_horizon $action_horizon \
    --inpaint_horizon $inpaint_horizon \
    --noise_steps $noise_steps \
    --cond_dim $cond_dim \
    --output_dim $output_dim \
    --model $model \
    --dataset_dir $dataset_dir \
    --dataset $dataset

model="UNet_Film"

python train.py \
    --n_epochs $n_epochs \

    --batch_size $batch_size \
    --lr $lr \
    --obs_horizon $obs_horizon \
    --pred_horizon $pred_horizon \
    --action_horizon $action_horizon \
    --inpaint_horizon $inpaint_horizon \
    --noise_steps $noise_steps \
    --cond_dim $cond_dim \
    --output_dim $output_dim \
    --model $model \
    --dataset_dir $dataset_dir \
    --dataset $dataset

obs_horizon=40
pred_horizon=40
action_horizon=1
inpaint_horizon=5

python train.py \
    --n_epochs $n_epochs \

    --batch_size $batch_size \
    --lr $lr \
    --obs_horizon $obs_horizon \
    --pred_horizon $pred_horizon \
    --action_horizon $action_horizon \
    --inpaint_horizon $inpaint_horizon \
    --noise_steps $noise_steps \
    --cond_dim $cond_dim \
    --output_dim $output_dim \
    --model $model \
    --dataset_dir $dataset_dir \
    --dataset $dataset

noise_steps=400
model="default"

python train.py \
    --n_epochs $n_epochs \

    --batch_size $batch_size \
    --lr $lr \
    --obs_horizon $obs_horizon \
    --pred_horizon $pred_horizon \
    --action_horizon $action_horizon \
    --inpaint_horizon $inpaint_horizon \
    --noise_steps $noise_steps \
    --cond_dim $cond_dim \
    --output_dim $output_dim \
    --model $model \
    --dataset_dir $dataset_dir \
    --dataset $dataset