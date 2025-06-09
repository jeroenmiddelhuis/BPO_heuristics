from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO
from rw_simulator import Simulator
from rw_gym_env import Environment
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import os
from stable_baselines3.common.callbacks import CheckpointCallback
import pandas as pd
from callbacks import PPOEvalCallback
import sys
    
def make_env(problem_name, nr_cases, action_setup="heuristics"):
    """Create and initialize the environment"""
    if problem_name == 'toloka':
        instance_file = "./data/toloka_problem.pkl"
    elif problem_name == 'fines':
        instance_file = "./data/fines_problem.pkl"
    elif problem_name == 'bpi2017':
        instance_file = "./data/bpi2017_problem.pkl"
        interarrival_rate_multiplier = 1
    elif problem_name == 'bpi2012':
        instance_file = "./data/bpi2012_problem.pkl"
        interarrival_rate_multiplier = 1
    elif problem_name == 'bpi2018':
        instance_file = "./data/bpi2018_problem.pkl"
        interarrival_rate_multiplier = 0.6
    elif problem_name == 'consulta':
        instance_file = "./data/consulta.pkl"
        interarrival_rate_multiplier = 2.3
    elif problem_name == 'production':
        instance_file = "data/production.pkl"
        interarrival_rate_multiplier = 1.8
    elif problem_name == 'microsoft':
        instance_file = "data/microsoft.pkl"
        interarrival_rate_multiplier = 1.1
    else:
        raise Exception("Invalid problem name")

    # Initialize simulator and environment
    if action_setup == "heuristics":
        simulator = Simulator(nr_cases=nr_cases, 
                              instance_file=instance_file, 
                              problem_name=problem_name, 
                              interarrival_rate_multiplier=interarrival_rate_multiplier)
        env = Environment(simulator)
    else:
        simulator = Simulator(nr_cases=nr_cases, 
                              instance_file=instance_file, 
                              problem_name=problem_name, 
                              interarrival_rate_multiplier=interarrival_rate_multiplier, 
                              reward_function="cycle_time")
        env = Environment(simulator, 
                          action_setup=action_setup)
    return env

def get_optimal_hyperparameters(action_setup="heuristics"):
    """
    Get the optimal hyperparameters for the MaskablePPO model.
    
    Args:
        action_setup: The action setup to use (default is "heuristics")
    
    Returns:
        A dictionary of optimal hyperparameters
    """
    if action_setup == "heuristics":
        return {
            'n_layers': 2,
            'n_neurons': 128,
            'n_steps': 16384,
            'batch_size': 512,
            'learning_rate': 0.003,
            'gamma': 1.0,
            'gae_lambda': 0.8726906492576301,
            'ent_coef': 0.04429657539113244,
            'n_epochs': 20
        }
    else:
        return {
            'n_layers': 2,
            'n_neurons': 128,
            'n_steps': 25600,
            'batch_size': 256,
            'learning_rate': 3e-05,
            'gamma': 0.999,
            'gae_lambda': 0.8726906492576301,
            'ent_coef': 0.04429657539113244,
            'n_epochs': 20
        }


def train_policy(problem_name, nr_cases=2500, total_timesteps=100000, action_setup="heuristics", plot=False):
    """
    Train a MaskablePPO policy
    
    Args:
        config_type: The configuration type to use for the simulator
        nr_cases: Number of cases to simulate
        total_timesteps: Total timesteps for training
        plot: Whether to plot policy usage and cycle time after training
    """
    print(f"Training policy for {problem_name} with {nr_cases} cases for {total_timesteps} timesteps")
    # Create the environment
    env = make_env(problem_name, nr_cases, action_setup)
    
    # Create a separate environment for evaluation
    eval_env = make_env(problem_name, nr_cases=2500, action_setup=action_setup)

    if action_setup == "assignments":
        save_path = f"models/PPO_assignments/{problem_name}/{problem_name}_final"
        best_model_path = f"models/PPO_assignments/{problem_name}/{problem_name}_best"
    else:
        # Create the model directory if it doesn't exist
        save_path = f"models/PPO/{problem_name}/{problem_name}_final"
        best_model_path = f"models/PPO/{problem_name}/{problem_name}_best"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Setup evaluation callback - evaluate every 10 updates
    eval_callback = PPOEvalCallback(
        eval_env=eval_env,
        eval_freq=10,
        n_eval_episodes=50,
        best_model_path=best_model_path,
        verbose=1
    )
    optimal_hyperparameters = get_optimal_hyperparameters(action_setup)

    # Build net_arch
    net_arch = dict(
        pi=[optimal_hyperparameters['n_neurons'] 
            for _ in range(optimal_hyperparameters['n_layers'])],
        vf=[optimal_hyperparameters['n_neurons'] 
            for _ in range(optimal_hyperparameters['n_layers'])]
    )

    # Create and train the model
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        n_steps=optimal_hyperparameters['n_steps'],
        batch_size=optimal_hyperparameters['batch_size'],
        learning_rate=optimal_hyperparameters['learning_rate'],
        gamma=optimal_hyperparameters['gamma'],
        gae_lambda=optimal_hyperparameters['gae_lambda'],
        ent_coef=optimal_hyperparameters['ent_coef'],
        n_epochs=optimal_hyperparameters['n_epochs'],
        policy_kwargs=dict(net_arch=net_arch)
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=False,
        callback=eval_callback
    )
    
    # Save the final model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Save evaluation results to CSV
    if action_setup == "heuristics":
        eval_callback.save_eval_results(problem_name)
    
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
    os.makedirs('data_training/', exist_ok=True)
    csv_path = f'data_training/{env.simulator.problem_name}_action_count.csv'
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
    plt.savefig(f'figures/training/{env.simulator.problem_name}_action_count.png')
    if show_plot:
        plt.show()

def main():
    # Train the policy
    problem_name = sys.argv[1] if len(sys.argv) > 1 else 'bpi2012'
    action_setup = 'assignments'
    model = train_policy(problem_name, nr_cases=1000, total_timesteps=1000, action_setup=action_setup, plot=True)
    # 
if __name__ == "__main__":
    main()

