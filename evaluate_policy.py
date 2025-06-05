from simulator import Simulator
from gym_env import Environment
import numpy as np
from heuristics import random_policy, spt_policy, fifo_policy
from heuristics import hrrn_policy, longest_queue_policy, shortest_queue_policy
from heuristics import least_flexible_resource_policy, most_flexible_resource_policy
from heuristics import least_flexible_activity_policy, most_flexible_activity_policy
import os
from sb3_contrib import MaskablePPO

def evaluate(config_type, policy, nr_cases, nr_episodes):
    # Initialize simulator and environment
    simulator = Simulator(config_type, nr_cases)
    env = Environment(simulator)
    
    cycle_times = []

    for _ in range(nr_episodes):
        env.reset()
        done = False
        step_count = 0

        while not done:
            action = policy(simulator)
            obs, reward, done, _, _ = env.step(action)
            step_count += 1

            # if step_count % 100 == 0:
            #     queue_lengths = simulator.get_queue_lengths_per_task_type()
            #     avg_queue_length = np.mean(list(queue_lengths.values()))
            #     print(f"Step {step_count}: Average queue length = {avg_queue_length:.2f}, Per task type: {queue_lengths}")

        assert len(simulator.completed_cases) == nr_cases, f"Expected {nr_cases} completed cases, but got {len(simulator.completed_cases)}"
        cycle_times.append(sum(case.cycle_time for case in simulator.completed_cases) / len(simulator.completed_cases))
        print(sum(case.cycle_time for case in simulator.completed_cases) / len(simulator.completed_cases))

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
    policies = [spt_policy, fifo_policy, random_policy, "postpone"]

    for _ in range(nr_episodes):
        # Main interaction loop
        obs, _ = env.reset()
        done = False

        while not done:
            action, probabilities = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            # ppo_assignment = policies[action](simulator)
            # spt_assignment = spt_policy(simulator)
            # if ppo_assignment != spt_assignment:
            #     print(f"Discrepancy found: PPO assignment {ppo_assignment} vs SPT assignment {spt_assignment}")

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
        f.write("cycle_time,spt_policy,fifo_policy,random_policy,postpone\n")
        for i, cycle_time in enumerate(cycle_times):
            f.write(f"{cycle_time},{track_actions[i]["spt_policy"]},{track_actions[i]["fifo_policy"]},{track_actions[i]["random_policy"]},{track_actions[i]["postpone"]}\n")


if __name__ == "__main__":
    # ['slow_server', 'parallel', 'low_utilization', 'high_utilization', 'down_stream', 'n_system', 'parallel_xor']:
    for config_type in ['n_system']:
        # Evaluate PPO policy
        # evaluate_ppo(config_type, nr_cases=2500, nr_episodes=300)
        

        #[random_policy, spt_policy, fifo_policy, shortest_queue_policy, longest_queue_policy]
        for policy in [least_flexible_resource_policy]:
            evaluate(config_type, policy, nr_cases=2500, nr_episodes=30)