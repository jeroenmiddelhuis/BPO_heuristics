from gymnasium import spaces, Env
import numpy as np
from heuristics import random_policy, spt_policy, fifo_policy


class Environment(Env):
    def __init__(self, simulator) -> None:
        super().__init__()
        self.simulator = simulator
        
        # Define action and observation spaces
        self.assignments = self.simulator.assignments
        self.actions = self.simulator.assignments + ["postone"]
        self.action_space = spaces.Discrete(len(self.actions))  # +1 for postpone

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.simulator.resources) + 
                                  len(self.assignments) + 
                                  len(self.simulator.task_types),), 
                                  dtype=np.float64)

    def reset(self, seed=None):
        """Reset the environment and return the initial observation."""
        self.simulator.reset()
        super().__init__()
        self.__init__(self.simulator)  # Reinitialize the environment
        
        # Run simulation until first decision point
        self.simulator.run_until_next_decision_epoch()
        
        return self.observation(), {}

    def step(self, action):
        """Execute one step in the environment."""        
        # The action is an integer and should be converted to a (resource, task) assignment
        if isinstance(action, (int, np.integer)):
            if action == len(self.actions) - 1:
                # Handle postpone action
                self.simulator.postpone()
            else:
                # Convert integer action to resource-task assignment
                resource, task_type = self.actions[action]
                assignment = self.simulator.resource_to_task_type_assignment(resource, task_type)
                
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
        if self.simulator.is_arrivals_coming():
            postpone_possible = np.array([1.0], dtype=np.float64)
        else:
            postpone_possible = np.array([0.0], dtype=np.float64)
        #postpone_possible = np.array([1.0 if not self.simulator.is_arrivals_coming() or sum(assignment_masks) < len(self.assignments) else 0.0], dtype=np.float64)
        return np.concatenate([assignment_masks, postpone_possible])

    def close(self):
        # Clean up resources
        pass

def main():
    pass

if __name__ == '__main__':
    main()