import json
import copy
import random

import numpy as np
import pandas as pd

from itertools import permutations, combinations
from abc import ABC, abstractmethod


def spt_policy(simulator):

    return

def fifo_policy(simulator):
    return 

def random_policy(simulator):
    return 



    


class Planner(ABC):
    """Abstract class that all planners must implement."""

    @abstractmethod
    def plan(self, available_resources, unassigned_tasks, resource_pool):
        """
        Assign tasks to resources from the simulation environment.

        :param environment: a :class:`.Simulator`
        :return: [(task, resource, moment)], where
            task is an instance of :class:`.Task`,
            resource is one of :attr:`.Problem.resources`, and
            moment is a number representing the moment in simulation time
            at which the resource must be assigned to the task (typically, this can also be :attr:`.Simulator.now`).
        """
        raise NotImplementedError


# Greedy assignment
class GreedyPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources in an anything-goes manner."""

    def plan(self, available_resources, unassigned_tasks, resource_pool):
        assignments = []
        available_resources = available_resources.copy()
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    available_resources.remove(resource)
                    assignments.append((task, resource))
                    break
        return assignments

    def report(self, event):
        pass#print(event)

class RandomPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources in an anything-goes manner."""

    def get_possible_assignments(self, available_resources, unassigned_tasks, resource_pool):
        possible_assignments = []
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    possible_assignments.append((resource, task))
        return possible_assignments
    
    def plan(self, available_resources, unassigned_tasks, resource_pool):
        available_resources = available_resources.copy()
        unassigned_tasks = unassigned_tasks.copy()        
        assignments = []

        possible_assignments = self.get_possible_assignments(available_resources, unassigned_tasks, resource_pool)

        #while len(possible_assignments) > 0:
        assignment = random.choice(possible_assignments)
        unassigned_tasks.remove(assignment[1])
        available_resources.remove(assignment[0])
        assignment = (assignment[0], assignment[1].task_type)
        assignments.append(assignment)


        return assignments

    def report(self, event):
        pass#print(event)

class ShortestProcessingTimeStandardized(Planner):
    def __init__(self) -> None:
        with open('distributions_standardized.json', 'r') as fp:
            self.distributions = json.load(fp)

    def linkSimulator(self, simulator):
        self.problem = simulator.problem
        distributions_or = simulator.problem.processing_time_distribution
        # Initialize the transformed dictionary
        self.distributions = {}

        # Iterate over the input dictionary
        for (main_key, sub_key), (first_value, _) in distributions_or.items():
            if main_key not in self.distributions:
                self.distributions[main_key] = {}
            self.distributions[main_key][sub_key] = first_value

        #standardize the distributions
        for key in self.distributions.keys():
            if key == 'Turning Rework':
                print(self.distributions[key])
            values = list(self.distributions[key].values())
            mean = np.mean(values)
            std = np.std(values)
            for sub_key in self.distributions[key].keys():
                if std != 0:
                    self.distributions[key][sub_key] = (self.distributions[key][sub_key] - mean) / std
                else:
                    self.distributions[key][sub_key] = mean

    def get_possible_assignments(self, available_resources, unassigned_tasks, resource_pool):
        possible_assignments = []
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    possible_assignments.append((resource, task))
        return possible_assignments
    
    def plan(self, available_resources, unassigned_tasks, resource_pool):
        available_resources = available_resources.copy()
        unassigned_tasks = unassigned_tasks.copy()        
        assignments = []

        possible_assignments = self.get_possible_assignments(available_resources, unassigned_tasks, resource_pool)

        while len(possible_assignments) > 0:

            spt = 999999
            best_assignment = None
            for assignment in possible_assignments: #assignment[0] = task, assignment[1]= resource
                if self.distributions[assignment[1].task_type][assignment[0]] < spt:
                    spt = self.distributions[assignment[1].task_type][assignment[0]]
                    best_assignment = assignment

            #check if best assignment is None
            if best_assignment is None:
                break
            unassigned_tasks.remove(best_assignment[1])
            available_resources.remove(best_assignment[0])
            best_assignment = (best_assignment[0], best_assignment[1].task_type)
            assignments.append(best_assignment)

            possible_assignments = self.get_possible_assignments(available_resources, unassigned_tasks, resource_pool)

        return assignments

    def report(self, event):
        pass#print(event)    


class ShortestProcessingTime(Planner):
    def __init__(self) -> None:
        #with open('distributions.json', 'r') as fp:
        #    self.distributions = json.load(fp)
        self.distributions = None

    def linkSimulator(self, simulator):
        self.problem = simulator.problem
        self.simulator = simulator
        distributions_or = simulator.problem.processing_time_distribution
        # Initialize the transformed dictionary
        self.distributions = {}

        # Iterate over the input dictionary
        for (main_key, sub_key), (first_value, _) in distributions_or.items():
            if main_key not in self.distributions:
                self.distributions[main_key] = {}
            self.distributions[main_key][sub_key] = first_value

    def get_possible_assignments(self, available_resources, unassigned_tasks, resource_pool):
        possible_assignments = []
        for task in unassigned_tasks:
            for resource in available_resources:
                if type(task) != str:
                    task = task.task_type
                if resource in resource_pool[task]:
                    possible_assignments.append((resource, task))
        return possible_assignments

    def get_possible_assignments_original(self, available_resources, unassigned_tasks, resource_pool):
        possible_assignments = []
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    possible_assignments.append((resource, task))
        return possible_assignments
    
    def plan(self, available_resources, unassigned_tasks, resource_pool):
        available_resources = available_resources.copy()
        unassigned_tasks = unassigned_tasks.copy()        
        assignments = []

        possible_assignments = self.get_possible_assignments_original(available_resources, unassigned_tasks, resource_pool)
        while len(possible_assignments) > 0:            
            spt = 999999
            for assignment in possible_assignments: #assignment[0] = task, assignment[1]= resource
                if self.distributions[assignment[1].task_type][assignment[0]] < spt:
                    spt = self.distributions[assignment[1].task_type][assignment[0]]
                    best_assignment = assignment
            
            unassigned_tasks.remove(best_assignment[1])
            available_resources.remove(best_assignment[0])
            best_assignment = (best_assignment[0], best_assignment[1].task_type)
            assignments.append(best_assignment)
            possible_assignments = self.get_possible_assignments_original(available_resources, unassigned_tasks, resource_pool)
        return assignments

    def report(self, event):
        pass#print(event)

    def plan_from_trace(self, available_resources, unassigned_tasks, resource_pool):
        available_resources = available_resources.copy()
        unassigned_tasks = unassigned_tasks.copy()
        assignments = []

        possible_assignments = self.get_possible_assignments(available_resources, unassigned_tasks, resource_pool)
        while len(possible_assignments) > 0:
            spt = 999999
            for assignment in possible_assignments:  # assignment[0] = task, assignment[1]= resource
                if self.distributions[assignment[1]][assignment[0]] < spt:
                    spt = self.distributions[assignment[1]][assignment[0]]
                    best_assignment = assignment

            unassigned_tasks.remove(best_assignment[1])
            available_resources.remove(best_assignment[0])
            best_assignment = (best_assignment[0], best_assignment[1])
            assignments.append(best_assignment)
            possible_assignments = self.get_possible_assignments(available_resources, unassigned_tasks, resource_pool)
        return assignments


class FIFOProcess(Planner):
    def __str__(self) -> str:
        return 'FIFO'

    def __init__(self):        
        self.resource_pools = None # passed through simulator
        self.task_types = None

    def get_possible_assignments(self, available_resources, unassigned_tasks, resource_pool):
        possible_assignments = []
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    possible_assignments.append((resource, task))
        return possible_assignments
    
    def plan(self, available_resources, available_tasks, resource_pools):
        available_tasks = available_tasks.copy()
        available_resources = available_resources.copy()        
        self.task_types = list(resource_pools.keys())

        assignments = []   
        case_priority_order = sorted(list(set([task.case_id for task in available_tasks]))) #cases are characterized by a monotonically increasing case_id
        priority_case = 0
        possible_assignments = self.get_possible_assignments(available_resources, available_tasks, resource_pools)
        while len(possible_assignments) > 0:
            priority_task_types = [task.task_type for task in available_tasks if task.case_id == case_priority_order[priority_case]]

            best_assignments = []
            while len(best_assignments) == 0:
                for possible_assignment in possible_assignments:
                    if possible_assignment[1].task_type in priority_task_types:
                        best_assignments.append((possible_assignment[0], possible_assignment[1].task_type))
                        possible_assignments.remove(possible_assignment)
                if len(best_assignments) == 0:
                    priority_case += 1
                    priority_task_types = [task.task_type for task in available_tasks if task.case_id == case_priority_order[priority_case]]        
            

        return best_assignments

class FIFOActivity(Planner):
    def __str__(self) -> str:
        return 'FIFO'

    def __init__(self):
        self.resource_pools = None  # passed through simulator
        self.task_types = None

    def get_possible_assignments(self, available_resources, unassigned_tasks, resource_pool):
        possible_assignments = []
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    possible_assignments.append((resource, task))
        return possible_assignments

    def plan(self, available_resources, available_tasks, resource_pools):
        pass