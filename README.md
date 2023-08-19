# Leveraging Human Intent for Shared Autonomy and Risk Negotiation
This project is part of the research of the research conducted by the [Assistive Robotics and Manipulation Lab](https://arm.stanford.edu/research/leveraging-human-intent-shared-autonomy). Please visit the website for the full scope of the project.

## Goal of this Project:

Our focus lies in exploring the feasibility of forecasting the upcoming trajectory of a human driver. This involves accumulating datasets comprising state (position , velocity), action, and environment (visual) information. Through this contextual information, we aim to predict not only vehicle trajectories but also predict human behavior action inputs (steering,acceleration and breaking).


## Project Description
In order to predict a vehicle trajectory it has to be determined what the human driver behaves. Furthermore humans differ, and their behavior can differ. Consequently, we view human driving behavior as a multimodal distribution. This conditions the distribution of the resulting trajectory, from which needs to be sampled from. 

We propose the use of conditioned diffusion models as a method of performing both of the previously mentioned steps. Diffusion models are able to model multimodal behavior, can easily introduce conditioning variables and sample directily from the conditioned trajectory distribution. 

![image](https://github.com/rafaelsoStanford/State_Policy_DiffusionModel/assets/130123073/d0f50da3-fe63-4367-8153-ebed1eab75d0)

##  How to Install
Clone the project in a directory of you choice:
```console
git clone <repository-url>
```
We recommend using a conda environment to manage project dependencies. Navigate to the cloned project's directory and run:
```console
conda create --name <env-name> python=3.9.0 --file requirements.txt
```
Replace <env-name> with the desired name for your conda environment.

>  We used Python 3.9.0, later versions could end up incompatible due to gym.
>  This repo relies on a modified carracing environment from gym v.21.0. Meaning it relies on the corresponding Box2d library.

## How to Run

### Generating Datasets

The datasets are generated using PID controllers to simulate various driving behaviors following different trajectories. These trajectories are controlled and deterministic, providing valuable ground truth data for the project.
Possible trajectories are:
    
- Parallel: Simulates vehicles driving on different sides of the road in a parallel manner.\
![SafeDriver](https://github.com/rafaelsoStanford/State_Policy_DiffusionModel/assets/130123073/ff83bbab-d324-405f-b2ab-8469a6ebfb3a)

- Sinusoidal: Mimics a drunk driving behavior with a sinusoidal trajectory.\
![SlalomDriverSafe](https://github.com/rafaelsoStanford/State_Policy_DiffusionModel/assets/130123073/b851ca09-a1ef-4f0d-99af-eb49c7f63393)


Navigate to the generateData folder using the following command:

```console
cd generateData
```

Once you're in the generateData folder, you can execute a generation script of your choice using the following command: 
```console
python <filename>.py. 
```
> If you need help or additional information about the available options, you can use the --help flag: `python filename.py --help`. 

After a successful run, you should find the dataset saved under `/data`.

### Training
Navigate to the base repository folder and run:
```console
python train.py --dataset="<Name-of-your-dataset>.zarr.zip"
```
> Make sure your dataset is saved at ./data, as the script assumes this path. \
> If you need help or additional information about the available options, you can use the `--help` flag

The trained model will learn from the dataset to make predictions. Depending on the flags you set, it can predict either trajectories, actions or a mixed output with both vectors stacked.

### Generate Predictions
In order to generate predictions you can run 
```console
python generate.py --dataset="<Name-of-your-dataset>.zarr.zip"
```


## Leveraging Human Intent for Shared Autonomy and Risk Negotiation

The diffusion model is designed to enable shared autonomy for transitions from AI to human control of a vehicle. The goal is to create an agent capable of identifying human driver behavior and safely transitioning vehicle control.

This project consists of three stages:

1. **Trajectory Prediction** - The agent utilizes a diffusion policy model trained on various human driver behaviors to make trajectory predictions conditioned on past state observations and driver inputs.

2. **Risk Assessment** - The agent employs risk negotiation strategies to assess potential risks associated with control transitions and makes proactive decisions to mitigate them. This ensures that the vehicle operates in a safe and reliable manner.

3. **Control Transition** - The system is designed to smoothly transfer control of the vehicle from the AI agent to the human driver. The transition should be smooth after evaluating the risks involved.

## Features

- **Behavior Identification**: The agent leverages a diffusion policy model trained on various human driver behaviors to predict trajectories based on past observations and driver inputs.

- **Risk Negotiation**: The agent employs strategies to assess potential risks during control transitions and makes proactive decisions to mitigate them, ensuring safe operation.

- **Control Transition**: The system facilitates smooth transfer of control from the AI agent to the human driver once risks have been evaluated and addressed.

- **Diffusion Model Training**: The repository includes a `train.py` file that can be used to train the diffusion policy using the `diffusion.py` file. The training process utilizes PyTorch Lightning.

- **Data Generation**: The data used for training can be generated using the files in the `generateData` folder. Specifically, there are two files that generate sinusoidal or parallel driving behavior.

- **Policy Generation**: The `generate.py` file can be used to generate policies based on the trained diffusion model.

## Prerequisites

```
...
```

## Files Tree

```
/home/rafael/git_repos/diffusion_bare/
├── ...
├── train.py
├── generateData
│   ├── generateParallelTraj.py
│   ├── generateSinusoidalTraj.py
│   ├── ...
├── generate.py
├── ...
```

- `train.py`: The main file for training the diffusion policy using the `diffusion.py` file with PyTorch Lightning.

- `generateParallelTraj.py`: File for generating parallel driving behavior data.

- `generateSinusoidalTraj.py`: File for generating sinusoidal driving behavior data.

- `generate.py`: File for generating policies based on the trained diffusion model.

Please note that the above list includes the main files relevant to the diffusion model. There may be additional files and directories in the project structure.
