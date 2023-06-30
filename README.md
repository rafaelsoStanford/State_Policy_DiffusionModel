# Diffusion Model - Robot Policies for Shared Autonomy

### WORK IN PROGRESS ###

</br>![](https://github.com/rafaelsoStanford/SharedAutonomy_RiskNegotiation/blob/AddGifs/files/SafeDriver.gif) 
</br>![](https://github.com/rafaelsoStanford/SharedAutonomy_RiskNegotiation/blob/AddGifs/files/SafeDriver.gif) 


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
