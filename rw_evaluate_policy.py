from rw_simulator import Simulator
from rw_gym_env import Environment
from rw_heuristics import random_policy, spt_policy, fifo_policy
from rw_heuristics import hrrn_policy, longest_queue_policy, shortest_queue_policy
from rw_heuristics import least_flexible_resource_policy, most_flexible_resource_policy
from rw_heuristics import least_flexible_activity_policy, most_flexible_activity_policy
import os
import numpy as np
import statistics


def evaluate(problem_name, policy, nr_cases, nr_episodes, write_to_file=False):
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
    simulator = Simulator(nr_cases=nr_cases, instance_file=instance_file, interarrival_rate_multiplier=interarrival_rate_multiplier)
    env = Environment(simulator)
    
    cycle_times = []
    all_min_cycle_times = []
    all_max_cycle_times = []
    all_ranges = []
    all_completed_tasks = []  # New list to store completed tasks count

    for episode in range(nr_episodes):
        env.reset()
        done = False
        step_count = 0

        while not done:
            action = policy(simulator)
            obs, reward, done, _, _ = env.step(action)
            step_count += 1
        print("running time:", simulator.now)
        #assert simulator.n_finalized_cases == nr_cases, f"Expected {nr_cases} completed cases, but got {simulator.n_finalized_cases}"
                        
        # Calculate cycle time statistics using the simulator's data
        avg_cycle_time = simulator.total_cycle_time / simulator.n_finalized_cases
        
        # Calculate min/max cycle times from case data
        case_cycle_times = simulator.case_cycle_times.values()
        
        min_cycle_time = min(case_cycle_times) if case_cycle_times else 0
        max_cycle_time = max(case_cycle_times) if case_cycle_times else 0
        cycle_time_range = max_cycle_time - min_cycle_time
        
        # Store metrics for later analysis
        cycle_times.append(avg_cycle_time)
        all_min_cycle_times.append(min_cycle_time)
        all_max_cycle_times.append(max_cycle_time)
        all_ranges.append(cycle_time_range)
        all_completed_tasks.append(simulator.total_completed_tasks)  # Store completed tasks count
        
        # Print a comprehensive summary for each episode
        print(f"Episode {episode+1}/{nr_episodes} statistics:")
        print(f"  Average cycle time: {avg_cycle_time:.2f}")
        print(f"  Min cycle time:     {min_cycle_time:.2f}")
        print(f"  Max cycle time:     {max_cycle_time:.2f}")
        print(f"  Range:              {cycle_time_range:.2f}")
        print(f"  Total cases:        {len(simulator.case_start_times)}")
        print(f"  Uncompleted cases:  {len(simulator.case_start_times)-len(simulator.case_cycle_times)}")
        print(f"  Completed tasks:    {simulator.total_completed_tasks}")

        print("-" * 50)
    
    # Calculate summary statistics across all replications
    avg_of_avgs = np.mean(cycle_times)
    std_of_avgs = np.std(cycle_times)
    avg_min = np.mean(all_min_cycle_times)
    avg_max = np.mean(all_max_cycle_times)
    avg_range = np.mean(all_ranges)
    avg_completed_tasks = np.mean([tasks/nr_cases for tasks in all_completed_tasks])  # Calculate average completed tasks
    std_completed_tasks = np.std([tasks/nr_cases for tasks in all_completed_tasks])   # Calculate std dev of completed tasks
    
    # Print summary statistics across all episodes
    print("\nSUMMARY STATISTICS ACROSS ALL EPISODES:")
    print(f"  Policy:                           {policy.__name__}")
    print(f"  Number of episodes:               {nr_episodes}")
    print(f"  Average cycle time:               {avg_of_avgs:.2f}")
    print(f"  Std dev of avg cycle times:       {std_of_avgs:.2f}")
    print(f"  95% CI for avg cycle time:        ({avg_of_avgs - 1.96*std_of_avgs/np.sqrt(nr_episodes):.2f}, {avg_of_avgs + 1.96*std_of_avgs/np.sqrt(nr_episodes):.2f})")
    print(f"  Average min cycle time:           {avg_min:.2f}")
    print(f"  Average max cycle time:           {avg_max:.2f}")
    print(f"  Average range:                    {avg_range:.2f}")
    print(f"  Average completed tasks per case: {avg_completed_tasks:.2f}")
    print(f"  Std dev of completed tasks:       {std_completed_tasks:.2f}")
    print("=" * 50)
    
    # Write results to file
    if write_to_file:
        results_dir = f'results/{problem_name}'
        os.makedirs(results_dir, exist_ok=True)
        results_file = f'{results_dir}/{problem_name}_{policy.__name__}.txt'
        with open(results_file, 'w') as f:
            f.write("cycle_time,completed_tasks\n")  # Add completed_tasks to header
            for i in range(len(cycle_times)):
                f.write(f"{cycle_times[i]},{all_completed_tasks[i]}\n")  # Write both metrics

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
    for problem_name in ['consulta']:
        for policy in [spt_policy]:
            results = evaluate(problem_name, policy, nr_cases=2500, nr_episodes=100, write_to_file=False)

            print(f"Policy: {policy.__name__}, Problem: {problem_name}")
            print(f"Avg Cycle Time (1000 cases): {results['avg_cycle_time']:.2f}")