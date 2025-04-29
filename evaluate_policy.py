from simulator import Simulator
from gym_env import Environment
import numpy as np
from heuristics import random_policy, spt_policy, fifo_policy

def main():
    # Initialize simulator and environment
    config_type = 'parallel'  # Change as needed
    nr_cases = 50000  # Change as needed
    simulator = Simulator(config_type, nr_cases)
    env = Environment(simulator)
    
    # Reset the environment
    obs, _ = env.reset()
    
    done = False
    total_reward = 0
    
    # Main interaction loop
    while not done:
        action = fifo_policy(simulator)
        
        # Take a step in the environment
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        # Print progress information
        #print(f"Time: {simulator.now:.2f}, Action: {action}, Reward: {reward:.4f}, Total Reward: {total_reward:.4f}")
    
    print(f"Simulation completed. Total reward: {total_reward:.4f}")
    print(f"Average cycle time: {sum(case.cycle_time for case in simulator.completed_cases) / len(simulator.completed_cases):.4f}")

if __name__ == "__main__":
    main()