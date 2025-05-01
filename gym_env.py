from gymnasium import spaces, Env
import numpy as np
from heuristics import random_policy, spt_policy, fifo_policy, shortest_queue_policy, longest_queue_policy
class Environment(Env):
    def __init__(self, simulator) -> None:
        super().__init__()
        self.simulator = simulator
        
        # Define action and observation spaces
        self.assignments = self.simulator.assignments
        self.actions = [spt_policy, fifo_policy, shortest_queue_policy, longest_queue_policy, "postpone"]
        self.track_actions = {policy.__name__ if callable(policy) else policy: 0 for policy in self.actions}
        
        # Add tracking for episode stats
        self.episode_count = 0
        self.stats_history = []
        
        self.action_space = spaces.Discrete(len(self.actions)) # +1 for postpone

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.simulator.resources) + 
                                  len(self.assignments) + 
                                  len(self.simulator.task_types),), 
                                  dtype=np.float64)

    def reset(self, seed=None):
        """Reset the environment and return the initial observation."""
        # Save stats from previous episode if it exists
        if self.episode_count > 0:
            avg_cycle_time = 0
            if self.simulator.completed_cases:
                avg_cycle_time = sum(case.cycle_time for case in self.simulator.completed_cases) / len(self.simulator.completed_cases)
            
            self.stats_history.append({
                'episode': self.episode_count,
                'actions': self.track_actions.copy(),
                'avg_cycle_time': avg_cycle_time
            })
            
        print(f"Episode {self.episode_count} completed. Action counts: {self.track_actions}")
        
        self.simulator.reset()
        # Reset action tracking but keep the episode count and stats history
        self.track_actions = {policy.__name__ if callable(policy) else policy: 0 for policy in self.actions}
        self.episode_count += 1
        
        # Run simulation until first decision point
        self.simulator.run_until_next_decision_epoch()
        
        return self.observation(), {}

    def step(self, action):
        """Execute one step in the environment."""        
        # The action is an integer and should be converted to a (resource, task) assignment
        if isinstance(action, (int, np.integer)):
            if action == len(self.actions) - 1:
                # Handle postpone action
                self.track_actions["postpone"] += 1
                self.simulator.postpone()
            else:
                # Convert integer action to resource-task assignment
                heuristic = self.actions[action]
                heuristic_name = heuristic.__name__ if callable(heuristic) else heuristic
                if heuristic_name in self.track_actions:
                    self.track_actions[heuristic_name] += 1

                assignment = heuristic(self.simulator)
                
                if assignment:
                    resource, task = assignment
                    # Process the action in simulator  
                    self.simulator.process_assignment(resource, task)

        else:
            # Use the tuple directly as the assignment
            resource, task = action
            self.simulator.process_assignment(resource, task)

        # Run simulation until next decision point or completion
        self.simulator.run_until_next_decision_epoch()
        
        # Additional info
        info = {}
        return self.observation(), self.get_reward(), self.is_done(), False, info  # False for truncated parameter

    def observation(self):
        # Resource features - add debug prints
        resource_features = np.array([resource.is_available() for resource in self.simulator.resources], dtype=np.float64)
        
        # Assignment features
        assignment_features = np.array([1.0 if resource.assigned_task is not None and 
                                        resource.assigned_task.task_type == task_type 
                                        else 0.0 
                                        for resource, task_type in self.assignments], 
                                        dtype=np.float64)

        # Task features
        task_features = np.array([sum(1.0 for task in self.simulator.get_ongoing_tasks() if task.task_type == task_type) / 100.0 
                    for task_type in self.simulator.task_types], dtype=np.float64)
        
        return np.concatenate([resource_features, assignment_features, task_features])

    def get_reward(self):
        reward = self.simulator.reward
        self.simulator.reward = 0
        return reward
    
    def is_done(self):
        return self.simulator.is_done()

    def render(self):
        pass

    def action_masks(self):
        assignment_masks = self.simulator.get_assignment_masks()
        if sum(assignment_masks) == 0:
            heuristic_mask = np.array([0.0] * (len(self.actions) - 1), dtype=np.float64)
        else:
            heuristic_mask = np.array([1.0] * (len(self.actions) - 1), dtype=np.float64)

        if self.simulator.is_arrivals_coming():
            postpone_possible = np.array([1.0], dtype=np.float64)
        else:
            postpone_possible = np.array([0.0], dtype=np.float64)

        return np.concatenate([heuristic_mask, postpone_possible])

    def close(self):
        # Clean up resources
        pass

def main():
    pass

if __name__ == '__main__':
    main()