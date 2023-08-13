execute_script() {
    echo "Starting execution of '$1'"
    start_time=$(date +"%Y-%m-%d %H:%M:%S")

    # Define the trap to handle the interrupt signal (Ctrl+C)
    trap 'handle_interrupt "$1" "$start_time"' INT

    python "$@"
    return_status=$?
    end_time=$(date +"%Y-%m-%d %H:%M:%S")
    if [ $return_status -eq 0 ]; then
        echo "Execution of '$1' successful"
        latest_folder=$(ls -td tb_logs/version_* | head -n 1)
        echo "Latest folder: $latest_folder"
        echo "$1: Success - Latest folder: $latest_folder (Started: $start_time, Ended: $end_time)" >> execution_summary.txt
    else
        if [ $return_status -eq 137 ]; then
            echo "Execution of '$1' killed"
            echo "$1: Killed (Started: $start_time, Ended: $end_time)" >> execution_summary.txt
        else
            echo "Execution of '$1' failed with status $return_status"
            echo "$1: Failed with status $return_status (Started: $start_time, Ended: $end_time)" >> execution_summary.txt
        fi
        exit $return_status
    fi

    # Extract the parsing options from the command
    parsing_options=$(echo "$@" | awk -F "--" '{print "--"$2}' | awk '{print $1}')

    # Append parsing options to the summary file
    echo "Parsing options: $parsing_options" >> execution_summary.txt
    echo "----------------------------------------" >> execution_summary.txt
}

handle_interrupt() {
    
    echo "***************************************" >> execution_summary.txt
    echo "Execution of '$1' interrupted"
    end_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "$1: Interrupted (Started: $2, Ended: $end_time)" >> execution_summary.txt
    echo "***************************************" >> execution_summary.txt
    exit 1
}


# Execute the Python scripts with different configurations
execute_script train.py --model="UNet_Film" --dataset="2023-08-07-0108_dataset_20_episodes_1_modes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=10 --inpaint_horizon=10 --step_size=5
# Execute the Python scripts with different configurations
execute_script train.py --model="UNet_Film" --dataset="2023-08-07-0108_dataset_20_episodes_1_modes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=20 --inpaint_horizon=10 --step_size=5
# Execute the Python scripts with different configurations
execute_script train.py --model="UNet_Film" --dataset="2023-08-07-0108_dataset_20_episodes_1_modes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=30 --inpaint_horizon=10 --step_size=1

execute_script train.py --model="UNet_Film" --dataset="2023-08-07-0108_dataset_20_episodes_1_modes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=10 --inpaint_horizon=5 --step_size=5
execute_script train.py --model="UNet_Film" --dataset="2023-08-07-0108_dataset_20_episodes_1_modes.zarr.zip" \
 --batch_size=16 --obs_horizon=10 --pred_horizon=20 --inpaint_horizon=5 --step_size=5
