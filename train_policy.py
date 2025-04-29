from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO
from simulator import Simulator
from gym_env import Environment
import numpy as np
import os
from stable_baselines3.common.callbacks import CheckpointCallback

def make_env(config_type, nr_cases):
    """Create and initialize the environment"""
    simulator = Simulator(config_type, nr_cases)
    env = Environment(simulator)
    return env

def train_policy(config_type, nr_cases=2500, total_timesteps=100000):
    """Train a MaskablePPO policy"""
    # Create the environment
    env = make_env(config_type, nr_cases)
    
    # Create the model directory if it doesn't exist
    save_path = "models/PPO/config_type"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Setup checkpoint callback to save models during training
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.dirname(save_path),
        name_prefix="maskable_ppo_checkpoint"
    )
    
    # Create and train the model
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        n_steps=2560,
        batch_size=256,
        tensorboard_log="./tensorboard_logs/"
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    # Save the final model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model

def main():
    # Train the policy
    model = train_policy(config_type='parallel', nr_cases=2500, total_timesteps=1000000)
    
    # Optional: Test the trained policy
    # env = make_env()
    # obs, _ = env.reset()
    
    # done = False
    # total_reward = 0
    
    # while not done:     
    #     # Get the action from the trained policy
    #     action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        
    #     # Take a step in the environment
    #     obs, reward, done, _, _ = env.step(action)
    #     total_reward += reward
    
    # print(f"Test evaluation - Total reward: {total_reward:.4f}")
    # print(f"Average cycle time: {sum(case.cycle_time for case in env.simulator.completed_cases) / len(env.simulator.completed_cases):.4f}")

if __name__ == "__main__":
    main()

