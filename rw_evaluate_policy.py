from rw_simulator import Simulator
from rw_gym_env import Environment
from rw_heuristics import random_policy, spt_policy, fifo_policy
from rw_heuristics import hrrn_policy, longest_queue_policy, shortest_queue_policy
from rw_heuristics import least_flexible_resource_policy, most_flexible_resource_policy
from rw_heuristics import least_flexible_activity_policy, most_flexible_activity_policy
import os
import numpy as np
import statistics
import sys
from sb3_contrib import MaskablePPO

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
        interarrival_rate_multiplier = 1.2
    elif problem_name == 'bpi2018':
        instance_file = "./data/bpi2018_problem.pkl"
        interarrival_rate_multiplier = 0.6
    elif problem_name == 'consulta':
        instance_file = "./data/consulta.pkl"
        interarrival_rate_multiplier = 2.5
    elif problem_name == 'production':
        instance_file = "data/production.pkl"
        interarrival_rate_multiplier = 2
    elif problem_name == 'microsoft':
        instance_file = "data/microsoft.pkl"
        interarrival_rate_multiplier = 1.3
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

def evaluate(problem_name, policy, nr_cases, nr_episodes, action_setup="heuristics", write_to_file=False):
    env = make_env(problem_name, nr_cases, action_setup)
    
    if policy == "ppo_policy":
        eval_policy_name = "ppo_policy"
        if action_setup == "heuristics":
            model = MaskablePPO.load(f"models/PPO/{problem_name}/{problem_name}_best.zip")
        elif action_setup == "assignments":
            model = MaskablePPO.load(f"models/PPO_assignments/{problem_name}/{problem_name}_best.zip")
        else:
            raise Exception(f"Unknown action setup: {action_setup}")
    else:
        eval_policy_name = policy.__name__ if callable(policy) else policy
    cycle_times = []
    all_min_cycle_times = []
    all_max_cycle_times = []
    all_ranges = []
    all_completed_tasks = []
    all_track_actions = []
    average_cases_in_system = []

    # Define the policies to track
    tracked_policies = [policy.__name__ for policy in env.actions if callable(policy)]

    # ["spt_policy", "fifo_policy", "hrrn_policy", 
    #                    "longest_queue_policy", "shortest_queue_policy", 
    #                    "least_flexible_resource_policy", "most_flexible_resource_policy",
    #                    "least_flexible_activity_policy", "most_flexible_activity_policy"]


    for episode in range(nr_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        track_actions = {policy_name: 0 for policy_name in tracked_policies}  # Initialize action counter
        
        while not done:
            if policy == "ppo_policy":
                action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
                action = np.int32(action)
                if action_setup == "heuristics":
                    # Track heuristic actions
                    heuristic = env.actions[action]
                    heuristic_name = heuristic.__name__ if callable(heuristic) else heuristic
                    track_actions[heuristic_name] += 1
            else:
                # Use the provided policy function to get the action
                action = policy(env.simulator)
            obs, reward, done, _, _ = env.step(action)
            step_count += 1
        print("running time:", env.simulator.now)
        #assert simulator.n_finalized_cases == nr_cases, f"Expected {nr_cases} completed cases, but got {simulator.n_finalized_cases}"
                        
        # Calculate cycle time statistics using the simulator's data
        avg_cycle_time = env.simulator.total_cycle_time / env.simulator.n_finalized_cases
        
        # Calculate min/max cycle times from case data
        case_cycle_times = env.simulator.case_cycle_times.values()
        
        min_cycle_time = min(case_cycle_times) if case_cycle_times else 0
        max_cycle_time = max(case_cycle_times) if case_cycle_times else 0
        cycle_time_range = max_cycle_time - min_cycle_time
        
        # Store metrics for later analysis
        cycle_times.append(avg_cycle_time)
        all_min_cycle_times.append(min_cycle_time)
        all_max_cycle_times.append(max_cycle_time)
        all_ranges.append(cycle_time_range)
        all_completed_tasks.append(env.simulator.total_completed_tasks)  # Store completed tasks count
        all_track_actions.append(track_actions)
        average_cases_in_system.append(env.simulator.avg_total_tasks)  # Store average cases in the system
        
        # Print a comprehensive summary for each episode
        print(f"Episode {episode+1}/{nr_episodes} statistics:")
        print(f"  Average cycle time: {avg_cycle_time:.2f}")
        print(f"  Min cycle time:     {min_cycle_time:.2f}")
        print(f"  Max cycle time:     {max_cycle_time:.2f}")
        print(f"  Range:              {cycle_time_range:.2f}")
        print(f"  Total cases:        {len(env.simulator.case_start_times)}")
        print(f"  Uncompleted cases:  {len(env.simulator.case_start_times)-len(env.simulator.case_cycle_times)}")
        print(f"  Completed tasks:    {env.simulator.total_completed_tasks}")
        print(f"  Average cases in the system: {env.simulator.avg_total_tasks:.2f}")

        print("-" * 50)
    
    # Calculate summary statistics across all replications
    avg_of_avgs = np.mean(cycle_times)
    std_of_avgs = np.std(cycle_times)
    avg_min = np.mean(all_min_cycle_times)
    avg_max = np.mean(all_max_cycle_times)
    avg_range = np.mean(all_ranges)
    avg_completed_tasks = np.mean([tasks/nr_cases for tasks in all_completed_tasks])  # Calculate average completed tasks
    std_completed_tasks = np.std([tasks/nr_cases for tasks in all_completed_tasks])   # Calculate std dev of completed tasks
    avg_of_average_cases_in_system = np.mean(average_cases_in_system)  # Average cases in the system
    
    # Print summary statistics across all episodes
    print("\nSUMMARY STATISTICS ACROSS ALL EPISODES:")
    print(f"  Policy:                           {eval_policy_name}")
    print(f"  Problem:                          {problem_name}")
    print(f"  Number of episodes:               {nr_episodes}")
    print(f"  Average cycle time:               {avg_of_avgs:.2f}")
    print(f"  Std dev of avg cycle times:       {std_of_avgs:.2f}")
    print(f"  95% CI for avg cycle time:        ({avg_of_avgs - 1.96*std_of_avgs/np.sqrt(nr_episodes):.2f}, {avg_of_avgs + 1.96*std_of_avgs/np.sqrt(nr_episodes):.2f})")
    print(f"  Average min cycle time:           {avg_min:.2f}")
    print(f"  Average max cycle time:           {avg_max:.2f}")
    print(f"  Average range:                    {avg_range:.2f}")
    print(f"  Average completed tasks per case: {avg_completed_tasks:.2f}")
    print(f"  Std dev of completed tasks:       {std_completed_tasks:.2f}")
    print(f"  Average cases in the system:      {avg_of_average_cases_in_system:.2f}")
    print("=" * 50)
    
    # Write results to file
    if write_to_file:
        results_dir = f'results/{problem_name}'
        os.makedirs(results_dir, exist_ok=True)
        if action_setup == "heuristics":
            results_file = f'{results_dir}/{problem_name}_{eval_policy_name}.txt'
        else:
            results_file = f'{results_dir}/{problem_name}_{eval_policy_name}_{action_setup}.txt'
        with open(results_file, 'w') as f:
            # Create header with all tracked policies
            header = "cycle_time"
            for policy_name in tracked_policies:
                header += f",{policy_name}"
            f.write(header + "\n")
            
            # Write data for each episode
            for i in range(len(cycle_times)):
                line = f"{cycle_times[i]}"
                for policy_name in tracked_policies:
                    line += f",{all_track_actions[i].get(policy_name, 0)}"
                f.write(line + "\n")

    return {
        'avg_cycle_time': avg_of_avgs,
        'std_dev': std_of_avgs,
        'cycle_times': cycle_times,
        'avg_completed_tasks': avg_completed_tasks,  # Add to return dictionary
        'std_dev_completed_tasks': std_completed_tasks,
        'completed_tasks': all_completed_tasks  # Return the list of completed tasks
    }

if __name__ == "__main__":
    # bpi2017, bpi2012, consulta, production, microsoft
    problem_name = sys.argv[1] if len(sys.argv) > 1 else 'production'
    policy_name = sys.argv[2] if len(sys.argv) > 2 else 'spt_policy'
    action_setup = sys.argv[3] if len(sys.argv) > 3 else 'heuristics'

    if policy_name == 'ppo_policy':
        policy = 'ppo_policy'
    elif policy_name == 'spt_policy':
        policy = spt_policy
    elif policy_name == 'fifo_policy':
        policy = fifo_policy
    elif policy_name == 'random_policy':
        policy = random_policy
    elif policy_name == 'hrrn_policy':
        policy = hrrn_policy
    elif policy_name == 'longest_queue_policy':
        policy = longest_queue_policy
    elif policy_name == 'shortest_queue_policy':
        policy = shortest_queue_policy
    elif policy_name == 'least_flexible_resource_policy':
        policy = least_flexible_resource_policy
    elif policy_name == 'most_flexible_resource_policy':
        policy = most_flexible_resource_policy
    elif policy_name == 'least_flexible_activity_policy':
        policy = least_flexible_activity_policy
    elif policy_name == 'most_flexible_activity_policy':
        policy = most_flexible_activity_policy
    else:
        raise Exception(f"Unknown policy name: {policy_name}")

    results = evaluate(problem_name, policy, nr_cases=2500, nr_episodes=300, action_setup=action_setup, write_to_file=False)