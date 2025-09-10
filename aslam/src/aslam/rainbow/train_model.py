#!/usr/bin/env python
from __future__ import division
from datetime import datetime
from tqdm import trange  # Provides progress bar for loops
from aslam.rainbow.pkgs.agent import Agent
from aslam.rainbow.pkgs.memory import ReplayMemory
from aslam.rainbow.pkgs.validate import validate
from gym.envs.registration import register

import bz2
import os
import pickle
import rospy
import numpy as np
import torch
from types import SimpleNamespace
import gym

########################################
# Utility Functions
########################################

def log(s):
    """Logs messages with ISO8601 timestamp."""
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
    """
    Loads a ReplayMemory object from disk.
    If disable_bzip is True, loads a plain pickle file.
    Otherwise, loads a bz2-compressed pickle file.
    """
    if disable_bzip:
        with open(memory_path, 'rb') as f:
            return pickle.load(f)
    else:
        with bz2.open(memory_path, 'rb') as f:
            return pickle.load(f)


def save_memory(memory, memory_path, disable_bzip):
    """
    Saves a ReplayMemory object to disk.
    If disable_bzip is True, saves as plain pickle.
    Otherwise, saves as bz2-compressed pickle.
    """
    if disable_bzip:
        with open(memory_path, 'wb') as f:
            pickle.dump(memory, f)
    else:
        with bz2.open(memory_path, 'wb') as f:
            pickle.dump(memory, f)
    log(f"Memory saves -> {memory_path}")


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

    # Access Resources folder to save checkpoints and results
    base_path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(base_path, 'Resources')
    
    results_path = os.path.join(resources_path, 'Results', args.exp_name)
    memory_path  = os.path.join(resources_path, 'Memory',  args.exp_name)
    model_path   = os.path.join(resources_path, 'Models',  args.exp_name)
    
    # Create folders if they don't exist
    for path in [results_path, memory_path, model_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Memory file path
    memory_file = os.path.join(memory_path, 'mem.bz2')

    # Initialize metrics dictionary for evaluation stats
    metrics = {'steps': [], 'cumulated_reward': [], 'explored_area': [], 'best_avg_reward': float('-inf')}

    args.model = None  # Optionally assign pretrained model later

    # Setting random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))

    # Setting device: use GPU if available and not disabled
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device('cpu')

    # Initialize gym environment
    register(id = args.gym_env,
            entry_point = 'aslam.openai_ros.task_envs.lilybot.lilybot_world:LilyBotWorldEnv',
            max_episode_steps = args.max_episode_length,)
    env = gym.make(args.gym_env)

    # Initialize agent and replay memory
    agent = Agent(args, env)
    if args.model is not None:
        mem = load_memory(memory_file, args.disable_bzip_memory)  # Loading existing memory
    else:
        mem = ReplayMemory(args, args.memory_capacity)  # Creating new memory

    # PER weight increase per step
    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

    ########################################
    # Training loop
    ########################################
    agent.train()  # Setting the agent to training mode
    done = True
    for T in trange(1, args.T_max + 1):
        if done:
            state = env.reset()  # Resetting environment at the start of each episode

        # Reset noisy weights periodically
        if T % args.replay_frequency == 0:
            agent.reset_noise()

        # Choose an action and perform one step
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        reward = max(min(reward, args.max_reward), args.min_reward)
        mem.append(state, action, reward, done)

        # Train Rainbow agent after learn_start steps
        if T >= args.learn_start:
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

            if T % args.replay_frequency == 0:
                agent.learn(mem)

            # Evaluate agent periodically
            if T % args.evaluation_interval == 0:
                agent.eval()
                reward = validate(args, T, env, agent, metrics, results_path, model_path)
                log(f"T = {T} / {args.T_max} | reward: {reward}")
                agent.train()
                done = True

            # Update target network periodically
            if T % args.target_update == 0:
                agent.update_target_net()

            # Save model checkpoint periodically
            if (args.model_checkpoint_interval != 0) and (T % args.model_checkpoint_interval == 0):
                agent.save(model_path, 'checkpoint.pth')

            # Save memory checkpoint periodically
            if (args.memory_checkpoint_interval != 0) and (T % args.memory_checkpoint_interval == 0):
                save_memory(mem, memory_file, args.disable_bzip_memory)

        state = next_state  # Moving to next state

