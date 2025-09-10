"""
Validation of Rainbow DQN agent performance during training.
Saves metrics, updates best models, and generates plots.
"""

import os
import numpy as np
import torch
import plotly.graph_objs as go
import plotly.offline as pyo


def validate(args, step, env, agent, metrics, results_dir, model_path):
    """
    Runs evaluation episodes to assess the performance of the agent.

    Args:
        args: Namespace of hyperparameters and config values.
        step (int): Current training step count.
        env: Gym-like environment.
        agent: Rainbow DQN agent.
        metrics (dict): Dictionary storing evaluation metrics.
        results_dir (str): Directory to save evaluation plots and metrics.
        model_path (str): Path to save the best model weights.

    Returns:
        float: Average cumulative reward across evaluation episodes.
    """
    
    metrics['steps'].append(step)

    cumulative_rewards = []
    explored_areas = []

    for _ in range(args.evaluation_episodes):
        state, done = env.reset(), False
        while not done:
            action = agent.act(state)  # Select greedy action from policy
            state, reward, done, _, _ = env.step(action)

        # Store episode results
        cumulative_rewards.append(env.cumulated_episode_reward)
        explored_areas.append(env.map_area)

    avg_cumulative_reward = float(np.mean(cumulative_rewards))
    avg_explored_area = float(np.mean(explored_areas))

    # Save model if performance improves
    if avg_cumulative_reward > metrics['best_avg_reward']:
        metrics['best_avg_reward'] = avg_cumulative_reward
        agent.save(model_path)

    # Log metrics
    metrics['cumulated_reward'].append(avg_cumulative_reward)
    metrics['explored_area'].append(avg_explored_area)

    torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

    # Generate performance plots
    _plot_metrics(
        steps=metrics['steps'],
        rewards=metrics['cumulated_reward'],
        areas=metrics['explored_area'],
        save_path=results_dir
    )

    return avg_cumulative_reward


def _plot_metrics(steps, rewards, areas, save_path=''):
    """
    Generates and saves interactive HTML plots for evaluation metrics.

    Args:
        steps (list[int]): Training steps corresponding to metrics.
        rewards (list[float]): Episodic cumulative rewards.
        areas (list[float]): Explored areas per episode.
        save_path (str): Directory path to save the plot.
    """
    # Define traces
    trace_reward = go.Scatter(
        x=steps,
        y=rewards,
        mode='lines',
        line=dict(color='rgb(0, 0, 180)'),
        name='Episodic Reward'
    )

    trace_area = go.Scatter(
        x=steps,
        y=areas,
        mode='lines',
        line=dict(color='rgb(0, 150, 0)'),
        name='Explored Area'
    )

    # Define layout
    layout = go.Layout(
        title='Evaluation Metrics',
        xaxis={'title': 'Training Step'},
        yaxis={'title': 'Metric Value'},
        template='plotly_white'
    )

    # Create figure
    fig = go.Figure(data=[trace_reward, trace_area], layout=layout)

    # Save HTML plot
    output_file = os.path.join(save_path, 'evaluation_metrics.html')
    pyo.plot(fig, filename=output_file, auto_open=False)

