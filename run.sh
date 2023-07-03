#!/bin/bash

# Remove the previous execution summary file
rm -f execution_summary.txt

# Function to execute Python script and check the return status
execute_script() {
    echo "Starting execution of '$1'"
    python "$@"
    return_status=$?
    if [ $return_status -eq 0 ]; then
        echo "Execution of '$1' successful"
        latest_folder=$(ls -td tb_logs/version_* | head -n 1)
        echo "Latest folder: $latest_folder"
        echo "$1: Success - Latest folder: $latest_folder" >> execution_summary.txt
    else
        if [ $return_status -eq 137 ]; then
            echo "Execution of '$1' killed"
            echo "$1: Killed" >> execution_summary.txt
        else
            echo "Execution of '$1' failed with status $return_status"
            echo "$1: Failed with status $return_status" >> execution_summary.txt
        fi
        exit $return_status
    fi
}




# Execute the Python scripts with different configurations
execute_script train.py --model="UNet_Film" --dataset="Sinusoidal_dataset_5_episodes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=20 --inpaint_horizon=5

execute_script train.py --model="UNet_Film" --dataset="Sinusoidal_dataset_5_episodes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=20 --inpaint_horizon=10

execute_script train.py --model="UNet_Film" --dataset="Sinusoidal_dataset_5_episodes.zarr.zip" \
 --batch_size=16 --obs_horizon=20 --pred_horizon=40 --inpaint_horizon=10

execute_script train.py --model="UNet_Film" --dataset="TestingActions_dataset_5_episodes_3_modes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=20 --inpaint_horizon=5

execute_script train.py --model="UNet_Film" --dataset="TestingActions_dataset_5_episodes_3_modes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=20 --inpaint_horizon=10

execute_script train.py --model="UNet_Film" --dataset="TestingActions_dataset_5_episodes_3_modes.zarr.zip" \
 --batch_size=16 --obs_horizon=20 --pred_horizon=40 --inpaint_horizon=10


execute_script train.py --model="UNet_Film" --dataset="Sinusoidal_dataset_5_episodes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=20 --inpaint_horizon=10 --noise_scheduler='linear_v2'

execute_script train.py --model="UNet_Film" --dataset="Sinusoidal_dataset_5_episodes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=20 --inpaint_horizon=10 --noise_scheduler='linear'
