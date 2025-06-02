import numpy as np
import random
from enum import Enum, auto
import json
from heuristics import random_policy, spt_policy, fifo_policy

class Event:
    def __init__(self, event_type, time, case=None, task=None, resource=None):
        self.event_type = event_type
        self.time = time
        self.case = case
        self.task = task
        self.resource = resource

    def __lt__(self, other):
        return self.time < other.time
    
class EventType(Enum):
    CASE_ARRIVAL = auto()
    TASK_COMPLETION = auto()
    CASE_DEPARTURE = auto()
    TAKE_ACTION = auto()
    
class Case:
    def __init__(self, case_id, arrival_time):
        self.case_id = case_id
        self.arrival_time = arrival_time
        self.cycle_time = None
        self.departure_time = None
        self.ongoing_tasks = [] # Ongoing tasks in the process, but not being executed yet
        self.waiting_tasks = [] # Waiting tasks have been executed, but are waiting for another task to be executed before they generate a new task
        self.executing_tasks = [] # Tasks that are being executed by a resource
        self.prefix = Prefix()

    def add_task(self, task):
        self.ongoing_tasks.append(task)

    def remove_task(self, task):
        self.ongoing_tasks.remove(task)

    def execute_task(self, task, now=None):
        self.remove_task(task)
        self.executing_tasks.append(task)
        task.start_time = now

    def complete_task(self, task, now=None):
        self.executing_tasks.remove(task)
        task.completion_time = now
        self.prefix.add_task(task)
    
class Task:
    def __init__(self, case_id, task_type, task_id=None, arrival_time=None):
        self.case_id = case_id
        self.task_type = task_type
        self.task_id = task_id
        self.arrival_time = arrival_time
        self.start_time = None
        self.completion_time = None

class Prefix:
    def __init__(self):
        self.prefix = []

    def add_task(self, task):
        self.prefix.append(task)

    def remove_task(self, task):
        self.prefix.remove(task)

    def get_prefix(self):
        return [task.task_type for task in self.prefix]
    
    def encode_prefix(self, prefix, task_types, encoding='one-hot'):
        encoded_prefix = []
        for task in prefix:
            if task.task_type == 'start':
                continue
            if encoding == 'one-hot':
                one_hot = [1 if task.task_type == t else 0 for t in task_types]
                encoded_prefix.append(one_hot)
        return encoded_prefix

class Resource:
    def __init__(self, resource_id):
        self.resource_id = resource_id
        self.eligibility = []
        self.processing_times = {}
        self.available = True
        self.assigned_task = None

    def is_available(self):
        return self.available

    def allocate(self, task):
        if not self.available:
            raise Exception(f"Resource {self.resource_id} is not available.")
        self.available = False
        self.assigned_task = task

    def release(self):
        if self.available:
            raise Exception(f"Resource {self.resource_id} is already available.")
        self.available = True
        self.assigned_task = None

class Simulator:
    def __init__(self, config_type, nr_cases, reward_function="cycle_time", print_results=False):
        self.config_type = config_type
        self.nr_cases = nr_cases
        self.reward_function = reward_function
        self.print_results = print_results

        self.events = []
        self.now = 0
        self.ongoing_cases = []
        self.completed_cases = []
        self.total_arrivals = 0
        self.total_generated_tasks = 0
        
        self.read_config(self.config_type)

        # Reinforcement learning parameters
        self.reward = 0
        self.last_reward_update = 0

        # Initialize the first case arrival
        self.events.append(Event(EventType.CASE_ARRIVAL, self.sample_interarrival_time()))

    def read_config(self, config_type):
        # Simulation parameters
        with open("config.txt", "r") as f:
            config_file = json.loads(f.read())
            config = config_file[config_type]

        self.arrival_rate = config["arrival_rate"]
        self.task_types_start_end = config["task_types"]
        self.task_types = config["task_types"][1:-1]        
        self.resources = []
        for r in config["resources"]:
            _resource = Resource(r)
            _resource.eligibility = config["resource_pools"][r].keys()
            _resource.processing_times = config["resource_pools"][r]
            self.resources.append(_resource)
        # Process structure with outputs converted to lists
        self.process_structure = config["process_structure"]
        # Convert comma-separated strings in outputs to tuples
        for task_type, structure in self.process_structure.items():
            new_outputs = {}
            for output_str, probability in structure["outputs"].items():
                if output_str:  # Skip empty strings
                    output_list = output_str.split(",")
                    new_outputs[tuple(output_list)] = probability
                else:
                    new_outputs[tuple()] = probability
            structure["outputs"] = new_outputs
        # Generate possible resource to task_type assignments
        self.assignments = []
        for task_type in self.task_types:
            for resource in self.resources:
                if task_type in resource.eligibility:
                    self.assignments.append((resource, task_type))

    def run_until_next_decision_epoch(self):
        """Run the simulator until a decision needs to be made or simulation is done."""
        while not self.is_done():
            event = self.events.pop(0)
            self.now = event.time

            if event.event_type == EventType.TAKE_ACTION:
                if self.reward_function == "AUC":
                    self.reward += -len(self.ongoing_cases) * (self.now - self.last_reward_update)
                    self.last_reward_update = self.now
                return True  # Decision point reached, return control to gym
            
            if event.event_type == EventType.CASE_ARRIVAL:
                if self.reward_function == "AUC":
                    self.reward += -len(self.ongoing_cases) * (self.now - self.last_reward_update)
                    self.last_reward_update = self.now
                case = self.generate_case()
                self.ongoing_cases.append(case)
                self.generate_tasks(case)
                if self.is_arrivals_coming():
                    self.events.append(Event(EventType.CASE_ARRIVAL, self.now + self.sample_interarrival_time()))
                if self.get_possible_assignments():
                    self.events.append(Event(EventType.TAKE_ACTION, self.now))

            elif event.event_type == EventType.TASK_COMPLETION:
                event.resource.release()
                event.case.complete_task(event.task, self.now)
                self.generate_tasks(event.case, event.task)
                # Check if the task is an end task
                if len(event.case.ongoing_tasks) == 1 and event.case.ongoing_tasks[0].task_type == 'end':
                    event.case.ongoing_tasks = [] # Remove the 'end' task
                    self.events.append(Event(EventType.CASE_DEPARTURE, self.now, case=event.case))
                # Check if there are any tasks that can be executed
                if self.get_possible_assignments():
                    self.events.append(Event(EventType.TAKE_ACTION, self.now))

            elif event.event_type == EventType.CASE_DEPARTURE:
                if self.reward_function == "AUC":
                    self.reward += -len(self.ongoing_cases) * (self.now - self.last_reward_update)
                    self.last_reward_update = self.now
                event.case.departure_time = self.now
                event.case.cycle_time = event.case.departure_time - event.case.arrival_time
                if self.reward_function == "cycle_time":
                    self.reward += 1 / (1 + event.case.cycle_time)
                self.ongoing_cases.remove(event.case)
                self.completed_cases.append(event.case)
                
            self.events.sort()
        
        if self.print_results:
            print("Simulation completed.")
            print("Total cycle time:", sum(case.cycle_time for case in self.completed_cases) / len(self.completed_cases))
            print("Total number of cases:", len(self.completed_cases))
            print("Total number of tasks:", self.total_generated_tasks)
            print("Total number of resources:", len(self.resources))
        return False  # Simulation is done

    def reset(self):
        """Reset the simulator to its initial state."""
        self.__init__(self.config_type, self.nr_cases, self.reward_function, self.print_results)

    def process_assignment(self, resource, task):
        """Process a single action by taking the TAKE_ACTION event and performing the action."""     
        # Execute the action
        resource.allocate(task)

        case = self.get_case(task.case_id)
        case.execute_task(task, self.now)
        
        # Create task completion event but DO NOT release the resource yet
        self.events.append(Event(EventType.TASK_COMPLETION, self.now + self.sample_processing_time(resource, task),
                            case=case, task=task, resource=resource))
        
        # Immediately add a TAKE_ACTION event if there are other possible assignments
        if self.get_possible_assignments():
            self.events.append(Event(EventType.TAKE_ACTION, self.now))
            
        self.events.sort()

    def postpone(self):
        """Skip the current decision point."""
        # Skip the event
        # Note that no new take_action event is added here, as it should be triggered by the simulator
        return 

    def is_done(self):
        # Check if all ongoing cases have completed tasks and there are no events left
        return len(self.ongoing_cases) == 0 and not self.is_arrivals_coming()
    
    def is_arrivals_coming(self):
        # Check if there are any arrivals left to process
        return self.total_arrivals < self.nr_cases

    def sample_interarrival_time(self):
        return random.expovariate(self.arrival_rate)
    
    def sample_processing_time(self, resource, task):
        return random.expovariate(1/resource.processing_times[task.task_type])

    def generate_case(self):
        case = Case(self.total_arrivals, self.now)
        self.total_arrivals += 1
        return case
    
    def get_case(self, case_id):
        for case in self.ongoing_cases:
            if case.case_id == case_id:
                return case
        return None
    
    def get_resource(self, resource_id):
        for resource in self.resources:
            if resource.resource_id == resource_id:
                return resource
        return None
    
    def generate_tasks(self, case, completed_task=None):
        """
        Generate next tasks based on the process structure.
        If a task was just completed, check if any new tasks can be started.
        When first called, completed_task is None (initial tasks from 'start')
        We assume that the process model is sound (i.e., no deadlocks or livelocks).
        """
        # Initial call - start from the "start" task
        if completed_task is None:
            completed_task = Task(case.case_id, 'start', None, self.now)
            case.prefix.add_task(completed_task)

        assert completed_task.case_id == case.case_id, "Task case ID does not match case ID"
        case.waiting_tasks.append(completed_task)

        # Sample next tasks based on process structure probabilities
        outputs = self.process_structure[completed_task.task_type]["outputs"]
        task_types, probabilities = zip(*outputs.items())
        next_task_types = np.random.choice(len(task_types), p=probabilities)
        next_task_types = task_types[next_task_types]

        # Check if next tasks can be generated (all input requirements are met)
        requirements_met = all(
            # For each next_task_type
            # Check if any of its input requirements are met
            any(
                # For each requirement in the inputs list
                # Check if this is a comma-separated requirement (AND logic)
                req.find(',') != -1 and all(
                    any(task.task_type == sub_req.strip() for task in case.waiting_tasks)
                    for sub_req in req.split(',')
                ) or
                # Regular requirement (OR logic across multiple requirements)
                req.find(',') == -1 and any(task.task_type == req for task in case.waiting_tasks)
                for req in self.process_structure[next_task_type]["inputs"]
            )
            for next_task_type in next_task_types
        )

        # Generate next tasks if all requirements are met
        if not requirements_met:
            # If requirements are not met, do not generate new tasks
            # The completed tasks is already added to waiting tasks
            return
        else:
            # Remove completed task(s) from waiting tasks based on task_type (req)
            for req in self.process_structure[completed_task.task_type]["inputs"]:
                for task in case.waiting_tasks:
                    if task.task_type == req:
                        case.waiting_tasks.remove(task)
                        break
            # Generate next tasks
            for next_task_type in next_task_types:
                if next_task_type != 'end':
                    task = Task(case.case_id, next_task_type, self.total_generated_tasks, self.now)
                    self.total_generated_tasks += 1
                else:
                    task = Task(case.case_id, next_task_type, None, self.now)
                    case.prefix.add_task(task)
                case.add_task(task)

    def get_ongoing_tasks(self):
        return [task for case in self.ongoing_cases for task in case.ongoing_tasks]
    
    def get_available_resources(self):
        return [resource for resource in self.resources if resource.is_available()]
    
    def get_ongoing_assignments(self):
        return 

    def get_possible_assignments(self):
        return [(resource, task) for task in self.get_ongoing_tasks() 
                for resource in self.get_available_resources() 
                if task.task_type in resource.eligibility]
    
    def get_queue_lengths_per_task_type(self):
        """
        Returns a dict mapping each task_type to the number of ongoing tasks of that type (queue length).
        """
        queue_lengths = {task_type: 0 for task_type in self.task_types}
        for case in self.ongoing_cases:
            for task in case.ongoing_tasks:
                if task.task_type in queue_lengths:
                    queue_lengths[task.task_type] += 1
        return queue_lengths

    def get_possible_resource_to_task_type_assignments(self):
        """
        Get all possible assignments of resources to task types that are currently ongoing.
        Returns list of (resource, task_type) tuples for available resources eligible for task types.
        """
        # Get unique task types from ongoing tasks
        ongoing_tasks = self.get_ongoing_tasks()
        available_resources = self.get_available_resources()
        if not ongoing_tasks or not available_resources:  # Early return if no ongoing tasks
            return []
            
        # Get unique task types from ongoing tasks
        task_types = {task.task_type for task in ongoing_tasks}
        
        # Filter the precomputed assignments based on currently available resources and task types
        return [(resource, task_type) 
                for resource, task_type in self.assignments
                if resource in available_resources and task_type in task_types]
        
    def get_assignment_masks(self):
        """
        Returns a binary mask (NumPy array) indicating which assignments in self.assignments
        are currently possible, based on get_possible_resource_to_task_type_assignments().
        """
        possible = set(self.get_possible_resource_to_task_type_assignments())
        return np.array([1.0 if assignment in possible else 0.0 for assignment in self.assignments], dtype=np.float64)

    def resource_to_task_type_assignment(self, resource, task_type):
        """
        Get the task of the case with the lowest task ID that can be assigned to the resource.
        """
        # First check if resource is eligible for this task type before processing tasks
        if task_type not in resource.eligibility:
            raise Exception(f"Resource {resource.resource_id} is not eligible for task type {task_type}.")
        
        if not resource.is_available():
            raise Exception(f"Resource {resource.resource_id} is not available.")
        
        # Find all tasks of the specified type using list comprehension
        matching_tasks = [task for task in self.get_ongoing_tasks() 
                        if task.task_type == task_type]
        
        if not matching_tasks:
            return None
        
        # Find the task with minimum task_id
        min_task = min(matching_tasks, key=lambda task: task.task_id)    
        return (resource, min_task)

if __name__ == "__main__":
    simulator = Simulator(config_type='parallel', nr_cases=5000)
    simulator.run(policy=fifo_policy)
    # case = simulator.generate_case()
    # simulator.ongoing_cases.append(case)
    # simulator.generate_tasks(case)
    # print([task.task_type for task in simulator.ongoing_cases[0].ongoing_tasks])
    # simulator.generate_tasks(case, simulator.ongoing_cases[0].ongoing_tasks[1])
    # print([task.task_type for task in simulator.ongoing_cases[0].ongoing_tasks])
    # simulator.generate_tasks(case, simulator.ongoing_cases[0].ongoing_tasks[0])
    # print([task.task_type for task in simulator.ongoing_cases[0].ongoing_tasks])
    # print("Simulation completed.")