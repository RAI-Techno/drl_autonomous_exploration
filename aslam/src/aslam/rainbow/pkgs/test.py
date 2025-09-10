"""
Testing of a pretrained Rainbow DQN agent.
Outputs exploration metrics.
"""

import os
import numpy as np
import torch


def test(args, env, agent):
    """
    Runs evaluation episodes to assess the performance of a pretrained agent.

    Args:
        args:  Namespace of hyperparameters and config values.
        env:   Gym-like environment used for evaluation.
        agent: Rainbow DQN agent.
    """

    successful_episodes = 0      # Number of episodes where exploration was successfuly completed
    N_collision         = 0      # Number of episodes that termiated due to collision
    exploration_times   = []     # Stores the exploration times of successful episodes
    trajectory_lengths  = []     # Stores the trajectory lengths of successful episodes

    for ep in range(1, args.evaluation_episodes + 1):
        state, done = env.reset(), False
        while not done:
            action = agent.act(state)  # Select greedy action from policy
            state, reward, done, _, _ = env.step(action)
            env.update_trajectory_length()

        env.calculate_exploration_time()
        # get and print exploration metrics
        print('#'*5,' Episode : ', ep, ' ', '#'*5)
        print('Termination reason : ', env.termination_reason)
        print('Trajectory length [m]   : ', env.trajectory_length)
        print('Exploration time  [sec] : ', env.exploration_time)
        print('')
        if(env.termination_reason == 'Exploration_Done'):
            exploration_times.append(env.exploration_time)
            trajectory_lengths.append(env.trajectory_length)
            successful_episodes += 1
        if(env.termination_reason == 'Collision_Occured'):
            N_collision += 1

    avg_exploration_time  = float(np.mean(exploration_times))
    avg_trajectory_length = float(np.mean(trajectory_lengths))
    success_rate   = successful_episodes/args.evaluation_episodes*100.0  # The percentage of episodes where exploration was successfuly completed
    collision_rate = N_collision/args.evaluation_episodes*100.0          # The percentage of episodes that terminated due to collision
    
    print('#'*5,' Average performance ', '#'*5)
    print('Success rate : ', success_rate, '%')
    print('Collision rate : ', collision_rate, '%')
    print('Trajectory length [m]   : ', avg_trajectory_length)
    print('Exploration time  [sec] : ', avg_exploration_time)
    
