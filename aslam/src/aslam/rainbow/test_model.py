#!/usr/bin/env python
from __future__ import division
from datetime import datetime
from tqdm import trange  # Provides progress bar for loops
from aslam.rainbow.pkgs.agent import Agent
from aslam.rainbow.pkgs.test import test
from gym.envs.registration import register

import os
import rospy
import numpy as np
import torch
from types import SimpleNamespace
import gym

########################################
# Utility Functions
########################################

def load_params():
    """
    Loads all hyperparameters and configurations from ROS param server.
    Provides default values if parameters are not set.
    Returns a SimpleNamespace object (args) containing all parameters.
    """
    return SimpleNamespace(
        seed=rospy.get_param("seed", 42),                                  # Random seed
        disable_cuda=rospy.get_param("disable_cuda", False),               # Flag to disable GPU
        T_max=rospy.get_param("T_max", int(3e5)),                          # Total training steps
        max_episode_length=rospy.get_param("max_episode_length", 2000),    # Maximum steps per episode
        history_length=rospy.get_param("history_length", 3),               # Number of consecutive frames per state
        noisy_std=rospy.get_param("noisy_std", 0.5),                       # Noisy layer standard deviation
        atoms=rospy.get_param("atoms", 51),                                # Number of atoms in distributional DQN
        V_min=rospy.get_param("V_min", -100.0),                            # Minimum support value for distributional DQN
        V_max=rospy.get_param("V_max", 100.0),                             # Maximum support value for distributional DQN
        model=rospy.get_param("model", None),                              # Pretrained model path
        memory_capacity=rospy.get_param("memory_capacity", 50000),         # Replay memory capacity
        replay_frequency=rospy.get_param("replay_frequency", 1),           # Steps between learning updates
        priority_exponent=rospy.get_param("priority_exponent", 0.5),       # PER alpha
        priority_weight=rospy.get_param("priority_weight", 0.4),           # PER beta
        multi_step=rospy.get_param("multi_step", 3),                       # Multi-step n value
        discount=rospy.get_param("discount", 0.99),                        # Discount factor gamma
        target_update=rospy.get_param("target_update", 1000),              # Steps to update target network
        learning_rate=rospy.get_param("learning_rate", 0.0000625),         # Adam learning rate
        adam_eps=rospy.get_param("adam_eps", 1.5e-4),                      # Adam epsilon
        batch_size=rospy.get_param("batch_size", 64),                      # Batch size
        norm_clip=rospy.get_param("norm_clip", 10.0),                      # Gradient clipping
        enable_cudnn=rospy.get_param("enable_cudnn", False),               # Flag to enable cuDNN
        disable_bzip_memory=rospy.get_param("disable_bzip_memory", False), # Flag to disable memory compression
        evaluation_interval=rospy.get_param("evaluation_interval", 5000),  # Steps between evaluations
        evaluation_episodes=rospy.get_param("evaluation_episodes", 5),     # Number of evaluation episodes
        model_checkpoint_interval=rospy.get_param("model_checkpoint_interval", 5000),     # Steps to checkpoint model
        memory_checkpoint_interval=rospy.get_param("memory_checkpoint_interval", 10000),  # Steps to checkpoint memory
        learn_start=rospy.get_param("learn_start", 1000),                  # Steps before training starts
        max_reward=rospy.get_param("max_reward", 1),                       # maximum reward
        min_reward=rospy.get_param("min_reward", -30),                     # minimum reward
        exp_name=rospy.get_param("exp_name", "Exp1"),                      # Experiment folder name
        gym_env=rospy.get_param("gym_env", "LilyBotWorld-v0")              # Name of open-AI gym environment
    )


########################################
# Main execution
########################################
if __name__ == "__main__":

    # Initialize ROS node
    rospy.init_node('main_code', anonymous=True)

    # Load all parameters into args
    args = load_params()

    # Access the pretrained model file.
    base_path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(base_path, 'Resources')
    
    model_path   = os.path.join(resources_path, 'Models', args.exp_name)
    args.model   = os.path.join(model_path, 'checkpoint.pth')

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))

    # Set device: using GPU if available and not disabled
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device('cpu')

    # Setup gym environment

    register(id = args.gym_env,
            entry_point = 'aslam.openai_ros.task_envs.lilybot.lilybot_world:LilyBotWorldEnv',
            max_episode_steps = args.max_episode_length,)
    env = gym.make(args.gym_env)

    # Setup agent
    agent = Agent(args, env)

    ########################################
    # Testing
    ########################################
    agent.eval()  # Setting Rainbow agent to evaluation mode
    test(args, env, agent)
