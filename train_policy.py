from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO
from simulator import Simulator
from gym_env import Environment
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import os
from stable_baselines3.common.callbacks import CheckpointCallback
import pandas as pd
from callbacks import PPOEvalCallback
import sys


def make_env(config_type, nr_cases):
    """Create and initialize the environment"""
    simulator = Simulator(config_type, nr_cases)
    env = Environment(simulator)
    return env

def train_policy(config_type, nr_cases=2500, total_timesteps=100000, plot=False):
    """
    Train a MaskablePPO policy
    
    Args:
        config_type: The configuration type to use for the simulator
        nr_cases: Number of cases to simulate
        total_timesteps: Total timesteps for training
        plot: Whether to plot policy usage and cycle time after training
    """
    print(f"Training policy for {config_type} with {nr_cases} cases for {total_timesteps} timesteps")
    # Create the environment
    env = make_env(config_type, nr_cases)
    
    # Create a separate environment for evaluation
    eval_env = make_env(config_type, nr_cases)

    # Create the model directory if it doesn't exist
    save_path = f"models/PPO/{config_type}/{config_type}_final"
    best_model_path = f"models/PPO/{config_type}/{config_type}_best"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Setup evaluation callback - evaluate every 10 updates
    eval_callback = PPOEvalCallback(
        eval_env=eval_env,
        eval_freq=10,
        n_eval_episodes=10,
        best_model_path=best_model_path,
        verbose=1
    )

    # Create and train the model
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        n_steps=5120,
        batch_size=256,
        tensorboard_log="./tensorboard_logs/"
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=False,
        callback=eval_callback
    )
    
    # Save the final model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Generate plot if requested
    if plot:
        plot_policy_usage_and_cycle_time(env, show_plot=False)
    
    return model

def plot_policy_usage_and_cycle_time(env, show_plot=True):
    """
    Plot the policy usage and average cycle time across episodes with smoothing.
    Also saves the data to a CSV file.
    
    Args:
        env: The environment containing stats_history
    """
    if not env.stats_history:
        print("No statistics available to plot.")
        return
    
    # Extract data from stats history
    episodes = [stat['episode'] for stat in env.stats_history]
    cycle_times = [stat['avg_cycle_time'] for stat in env.stats_history]
    
    # Get all policy names
    policy_names = list(env.stats_history[0]['actions'].keys())
    
    # Extract policy usage counts
    policy_usage = {policy: [stat['actions'][policy] for stat in env.stats_history] for policy in policy_names}
    
    # Save data to CSV
    data = {'Episode': episodes, 'Avg_Cycle_Time': cycle_times}
    for policy, counts in policy_usage.items():
        data[f'{policy}_actions'] = counts
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    csv_path = f'data/{env.simulator.config_type}_action_count.csv'
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Define smoothing function using moving average
    def smooth(y, window_size=5):
        if len(y) < window_size:
            return y
        # Use pandas rolling with center=True to handle edges properly
        s = pd.Series(y)
        return s.rolling(window=window_size, center=True, min_periods=1).mean().values
    
    # Plot policy usage on the left axis with smoothing
    for policy, counts in policy_usage.items():
        # Plot original data with low alpha
        # ax1.plot(episodes, counts, alpha=0.3, marker='.', markersize=3)
        # Plot smoothed data
        smoothed_counts = smooth(counts, window_size=5)
        ax1.plot(episodes, smoothed_counts, label=f"{policy} actions")
    
    # Plot average cycle time on the right axis with smoothing
    # ax2.plot(episodes, cycle_times, alpha=0.3, color='red', marker='.', markersize=3)
    smoothed_cycle_times = smooth(cycle_times, window_size=5)
    ax2.plot(episodes, smoothed_cycle_times, color='red', 
             linestyle='--', label='Avg Cycle Time')
    
    # Set labels and title
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Action count')
    ax2.set_ylabel('Average cycle time')
    ax1.set_title('Action count and average cycle time by episode')
    
    # Set integer x-axis
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
               bbox_to_anchor=(0.5, -0.1), ncol=int((len(policy_names) + 1)/2))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname('figures/training'), exist_ok=True)
    plt.savefig(f'figures/training/{env.simulator.config_type}_action_count.png')
    if show_plot:
        plt.show()

def main():
    # Train the policy
    # ['parallel_xor', 'parallel', 'low_utilization', 'high_utilization', 'slow_server', 'down_stream', 'n_system']
    #for config_type in ['parallel_xor', 'parallel', 'low_utilization', 'slow_server', 'down_stream', 'n_system']:
    config_type = sys.argv[1] if len(sys.argv) > 1 else 'slow_server'
    model = train_policy(config_type, nr_cases=1000, total_timesteps=5000000, plot=True)
    # 
if __name__ == "__main__":
    main()

