# Self-driving-car-nd

Udacity's Self-Driving Car Nanodegree project files and notes.

This repository contains project files for Udacity's Self-Driving Car Nanodegree program which I started working on 14 December, 2017.

# Program Outline:

### Term 1: Deep Learning and Computer Vision
#### 1. Computer Vision
- Project 1: Finding Lane Lines (Intro to Computer Vision)
- Project 4: Advanced Lane Lines
- Project 5: Vehicle Detection

#### 2. Deep Learning
- Project 2: Traffic Sign Classifier
- Project 3: Behavioural Cloning
  - Train a car to drive in a 3D simulator using a deep neural network.
  - Input data comprises steering angles and camera images captured by driving with a keyboard / mouse / joystick in the simulator.


# Conda Environment

## Creating an environment

By default, environments are installed into the envs directory in your conda directory. 
```
conda create --name myenv python=x.y
```

Creating an environment from an environment.yml file
```
conda env create -f environment.yml
```
## Activating an environment
```
source activate myenv
```

## Viewing a list of your environments
```
conda env list
```

## Viewing a list of the packages in an environment
To see a list of all packages installed in a specific environment:
```
conda list -n myenv
```

## Opening the code in a Jupyter Notebook
```
jupyter notebook
```

