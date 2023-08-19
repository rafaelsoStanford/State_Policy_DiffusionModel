# Leveraging Human Intent for Shared Autonomy and Risk Negotiation
This project is part of the research of the research conducted by the [Assistive Robotics and Manipulation Lab](https://arm.stanford.edu/research/leveraging-human-intent-shared-autonomy). Please visit the website for the full scope of the project.

## Goal of this Project:

Our focus lies in exploring the feasibility of forecasting the upcoming trajectory of a human driver. This involves accumulating datasets comprising state (position , velocity), action, and environment (visual) information. Through this contextual information, we aim to predict not only vehicle trajectories but also predict human behavior action inputs (steering,acceleration and breaking).


## Project Description
In order to predict a vehicle trajectory it has to be determined what the human driver behaves. Furthermore humans differ, and their behavior can differ. Consequently, we view human driving behavior as a multimodal distribution. This conditions the distribution of the resulting trajectory, from which needs to be sampled from. 

We propose the use of conditioned diffusion models as a method of performing both of the previously mentioned steps. Diffusion models are able to model multimodal behavior, can easily introduce conditioning variables and sample directily from the conditioned trajectory distribution. 

![image](https://github.com/rafaelsoStanford/State_Policy_DiffusionModel/assets/130123073/d0f50da3-fe63-4367-8153-ebed1eab75d0)

##  How to Install and Run the Project
Clone the project in a directory of you choice:
'''bash
git clone
'''


We suggest to create a conda environment and install the requirements. Enter the repository and run:
'''bash
conda create --name <env> --file requirements.txt
'''





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
