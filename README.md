# Learning-based Active SLAM: Rainbow DQN agent with RTAB-Map for autonomous exploration in indoor environments
<p align="center">
  <a href="https://youtu.be/yGzYkTcRFN4" target="_blank">
    <img src="/aslam/gifs/sim.gif" width="680">
  </a>
</p>

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [Real-World Experiment](#Real-World-Experiment)
- [References](#references)
- [Authors](#Authors)
---

## Overview

  Autonomous exploration of indoor environments is a fundamental task in robotics, enabling robots to efficiently build maps of unknown spaces. This repository implements a **learning-based active SLAM system**, combining a decision-making Rainbow DQN agent with **RTAB-Map** that performs 2D and 3D mapping.  

  The implemented system utilizes **OpenAI ROS** to interface reinforcement learning algorithms with ROS environments. A **custom robot** equipped with a LIDAR and a RealSense RGB-D camera is used for the mission.  

  This repository provides the full training and evaluation scripts for the Rainbow DQN agent performing active SLAM in indoor environments.

---

## Requirements

### ROS & Simulator
- [ROS Noetic](http://wiki.ros.org/noetic/Installation)  
- [Gazebo Simulator](https://classic.gazebosim.org/download)  

### Python Modules
- [PyTorch](https://pytorch.org/get-started/locally/)  
- [NumPy](https://numpy.org/install/)  
- [Gym](https://www.gymlibrary.ml/)  
- [tqdm](https://pypi.org/project/tqdm/)  
- [plotly](https://plotly.com/python/getting-started/)  
- [OpenCV (cv2)](https://opencv.org/releases/)  

### ROS Packages
- All [RTAB-Map ROS packages](http://wiki.ros.org/rtabmap_ros)
- [Realsense2 Camera](https://wiki.ros.org/realsense2_camera)

---

## Installation Guide

clone this repository inside your ROS workspace by executing the following commands in `terminal`: 
```console
cd <your_workspace_directory>/src
git clone https://github.com/RAI-Techno/drl_autonomous_exploration.git
cd ..
catkin_make
source devel/setup.bash
```

## Usage

1. **Configure parameters using YAML files**  
Check out three configuration files in `aslam/config/`:
- `RtabMap.yaml` – RTAB-Map mapping parameters
- `RL.yaml` – Rainbow DQN hyperparameters
- `task.yaml` – Task-specific parameters such as action space, initial poses, and exploration thresholds

Feel free to edit these files to adjust parameters according to your environment.

2. **Launch the slam system (robot + RTAB-Map)**
```console
roslaunch lilybot aslam.launch
```

3. **Train the agent**

Open another terminal and run the training script using:
```console
roslaunch aslam start_training.launch
```

4. **Test the trained agent**

To evaluate the trained agent, launch the testing script with:
```console
roslaunch aslam start_testing.launch
```
## Real-World Experiment

<p align="center">
  <a href="https://youtu.be/vT4LzGtNdDk" target="_blank">
    <img src="/aslam/gifs/real_world.gif" width="680">
  </a>
</p>

## References

This work relies on prior open-source contributions:

- **[LilyBot](https://github.com/RAI-Techno/lilybot)**  
- **[openai_ros](http://wiki.ros.org/openai_ros)**  
- **[Rainbow](https://github.com/Kaixhin/Rainbow)**

## Authors
- [Ali Hasan](https://github.com/Ali-Hasan-617)
- [Bashar Moalla](https://github.com/basharmoalla)
- [Yousef Mahfoud](https://github.com/yousef4422)
