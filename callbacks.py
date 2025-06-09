from typing import Callable
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


from typing import Callable
import os
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


class PPOEvalCallback(BaseCallback):
    """
    Callback for evaluating and saving a PPO model.
    
    :param eval_env: The environment used for evaluation
    :param eval_freq: Number of model updates between evaluations (in terms of optimizer calls)
    :param n_eval_episodes: Number of episodes to evaluate the model on
    :param best_model_path: Path to save the best model
    :param verbose: Verbosity level: 0 for no output, 1 for evaluation info
    """
    def __init__(
        self,
        eval_env,
        eval_freq=10,
        n_eval_episodes=20,
        best_model_path="./models/best_model",
        verbose=1,
    ):
        super(PPOEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_path = best_model_path
        self.best_mean_cycle_time = float("inf")
        self.update_count = 0
        # Add storage for evaluation history
        self.eval_history = {
            'timesteps': [],
            'mean_cycle_time': []
        }
        
    def _init_callback(self):
        # Create folder if needed
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            
    def _on_step(self):
        # Check if we're ready for evaluation - only check every eval_freq * batch updates
        # In PPO, n_steps is the steps collected before update, so we need to account for this
        if self.num_timesteps % (self.eval_freq * self.model.n_steps) != 0:
            return True
        
        # Start evaluation
        if self.verbose > 0:
            print(f"\n----- Evaluating model at {self.num_timesteps} timesteps -----")
        
        # Collect cycle times over multiple episodes
        cycle_times = []
        for i in range(self.n_eval_episodes):
            # Reset the environment for each episode
            obs, _ = self.eval_env.reset()
            done = False
            
            # Run a complete episode
            steps = 0
            while not done:
                # Use deterministic actions for evaluation
                action, _ = self.model.predict(
                    obs, 
                    action_masks=self.eval_env.action_masks(),
                    deterministic=True
                )

                # Take step in the environment
                obs, reward, done, truncated, info = self.eval_env.step(np.int32(action))
                if truncated:
                    done = True
                steps += 1
                
            # Extract cycle time metrics from the completed episode
            if hasattr(self.eval_env.simulator, "completed_cases") and len(self.eval_env.simulator.completed_cases) > 0:
                episode_cycle_time = sum(case.cycle_time for case in self.eval_env.simulator.completed_cases) / len(self.eval_env.simulator.completed_cases)
                cycle_times.append(episode_cycle_time)
                if self.verbose > 0:
                    print(f"Episode {i+1}/{self.n_eval_episodes}: Cycle time = {episode_cycle_time:.2f}")
            elif hasattr(self.eval_env.simulator, "total_cycle_time") and self.eval_env.simulator.total_cycle_time > 0:
                # Fallback if no completed cases, use total cycle time
                episode_cycle_time = self.eval_env.simulator.total_cycle_time / self.eval_env.simulator.n_finalized_cases if self.eval_env.simulator.n_finalized_cases > 0 else 0
                cycle_times.append(episode_cycle_time)
                if self.verbose > 0:
                    print(f"Episode {i+1}/{self.n_eval_episodes}: Cycle time = {episode_cycle_time:.2f}")
            else:
                Exception("No completed cases or total cycle time available for evaluation.")
        
        # Calculate average cycle time
        if cycle_times:
            mean_cycle_time = np.mean(cycle_times)
            
            # Record evaluation results
            self.eval_history['timesteps'].append(self.num_timesteps)
            self.eval_history['mean_cycle_time'].append(mean_cycle_time)
            
            if self.verbose > 0:
                print(f"Mean cycle time over {self.n_eval_episodes} episodes: {mean_cycle_time:.2f}")
                print(f"Previous best mean cycle time: {self.best_mean_cycle_time:.2f}")
            
            # Save if better than previous best
            if mean_cycle_time < self.best_mean_cycle_time:
                self.best_mean_cycle_time = mean_cycle_time
                if self.verbose > 0:
                    print(f"New best mean cycle time: {mean_cycle_time:.2f}")
                    print(f"Saving new best model to {self.best_model_path}")
                self.model.save(self.best_model_path)
        
        return True
    
    def save_eval_results(self, config_type):
        """
        Save evaluation results to a CSV file
        
        Args:
            config_type: Configuration type used for the simulation
        """
        if not self.eval_history['timesteps']:
            print("No evaluation results to save.")
            return
            
        # Create DataFrame
        df = pd.DataFrame(self.eval_history)
        
        # Save to CSV
        os.makedirs('data', exist_ok=True)
        csv_path = f'data/{config_type}_eval_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Evaluation results saved to {csv_path}")

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func