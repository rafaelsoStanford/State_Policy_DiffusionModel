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
Using the Generate Data scripts, you can create a testing trajectory.
```console
python <filename>.py. --num_episodes_per_mode=1 --dataset_name=<Name-of-your-training-dataset> --modes=['middle']
```
Create an animations folder:
```console
mkdir animations
```

In order to generate predictions you can run 
```console
python generate.py --dataset="<Name-of-your-training-dataset>.zarr.zip"
```
> **Be sure to load a dataset which is not the one you used for training.**

You should generate similar animation displaying the denoising process of diffusion: 
- ### Denoising Position predictions

![ezgif-2-d597b316fe](https://github.com/rafaelsoStanford/State_Policy_DiffusionModel/assets/130123073/d79dccfc-5b39-49c8-b06a-646e04a8186f) \
[DDPM Sampling](https://arxiv.org/abs/2006.11239)

![ezgif-2-c74990b1a0](https://github.com/rafaelsoStanford/State_Policy_DiffusionModel/assets/130123073/b02db002-237d-4433-af4a-ef296ad2de73) \
[DDIM Sampling](https://arxiv.org/abs/2010.02502)



### Run Simulation

Finally you can run a Simulation where the trajectories are estimation inside of the environment and displayed inside of the renderer.


