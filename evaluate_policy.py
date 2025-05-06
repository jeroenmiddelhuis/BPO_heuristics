from simulator import Simulator
from gym_env import Environment
import numpy as np
from heuristics import random_policy, spt_policy, fifo_policy, longest_queue_policy, shortest_queue_policy
import os
from sb3_contrib import MaskablePPO

def evaluate(config_type, policy, nr_cases, nr_episodes):
    # Initialize simulator and environment
    simulator = Simulator(config_type, nr_cases)
    env = Environment(simulator)
    
    cycle_times = []

    for _ in range(nr_episodes):
        # Main interaction loop
        env.reset()
        done = False

        while not done:
            action = policy(simulator)
            
            # Take a step in the environment
            obs, reward, done, _, _ = env.step(action)
        
        assert len(simulator.completed_cases) == nr_cases, f"Expected {nr_cases} completed cases, but got {len(simulator.completed_cases)}"
        cycle_times.append(sum(case.cycle_time for case in simulator.completed_cases) / len(simulator.completed_cases))
        # print(f"Simulation completed. Total reward: {total_reward:.4f}")
        # print(f"Average cycle time: {sum(case.cycle_time for case in simulator.completed_cases) / len(simulator.completed_cases):.4f}")

    # Write results to file
    results_dir = f'results/{config_type}'
    os.makedirs(results_dir, exist_ok=True)
    results_file = f'{results_dir}/{config_type}_{policy.__name__}.txt'
    with open(results_file, 'w') as f:
        f.write("cycle_time\n")
        for cycle_time in cycle_times:
            f.write(f"{cycle_time}\n")

def evaluate_ppo(config, nr_cases, nr_episodes):
    # Load the PPO model
    model_path = f"models/PPO/{config}/{config}_best.zip"
    model = MaskablePPO.load(model_path)

    # Initialize simulator and environment
    simulator = Simulator(config, nr_cases)
    env = Environment(simulator)

    cycle_times = []
    track_actions = []
    actions = [spt_policy, fifo_policy, shortest_queue_policy, longest_queue_policy, "postpone"]

    for _ in range(nr_episodes):
        # Main interaction loop
        obs, _ = env.reset()
        done = False

        while not done:
            action, probabilities = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
            #print(f"Action probabilities: {probabilities}", f"Action: {action}")
            obs, reward, done, _, _ = env.step(np.int32(action))

        assert len(simulator.completed_cases) == nr_cases, f"Expected {nr_cases} completed cases, but got {len(simulator.completed_cases)}"
        cycle_times.append(sum(case.cycle_time for case in simulator.completed_cases) / len(simulator.completed_cases))
        track_actions.append(env.track_actions.copy())

    # Write results to file
    results_dir = f'results/{config}'
    os.makedirs(results_dir, exist_ok=True)
    results_file = f'{results_dir}/{config}_ppo.txt'
    with open(results_file, 'w') as f:
        f.write("cycle_time,spt_policy,fifo_policy,shortest_queue_policy,longest_queue_policy,postpone\n")
        for i, cycle_time in enumerate(cycle_times):
            f.write(f"{cycle_time},{track_actions[i]["spt_policy"]},{track_actions[i]["fifo_policy"]},{track_actions[i]["shortest_queue_policy"]},{track_actions[i]["longest_queue_policy"]},{track_actions[i]["postpone"]}\n")


if __name__ == "__main__":
    # ['slow_server', 'parallel', 'low_utilization', 'high_utilization', 'down_stream', 'n_system', 'parallel_xor']:
    for config_type in ['slow_server', 'parallel', 'low_utilization', 'high_utilization', 'down_stream', 'n_system', 'parallel_xor']:
        # Evaluate PPO policy
        evaluate_ppo(config_type, nr_cases=1000, nr_episodes=300)
        
        # for policy in [random_policy, spt_policy, fifo_policy, shortest_queue_policy, longest_queue_policy]:
        #     evaluate(config_type, policy, nr_cases=1000, nr_episodes=300)