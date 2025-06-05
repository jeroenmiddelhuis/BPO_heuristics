import json
import copy
import random

import numpy as np
import pandas as pd

from itertools import permutations, combinations
from abc import ABC, abstractmethod



def spt_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    # Print all possible assignments with their expected processing times
    # print("Possible assignments (Resource, Task Type) and their expected processing times:")
    # for assignment in possible_assignments:
    #     resource, task_type = assignment
    #     processing_time = simulator.problem.processing_time_distribution[(task_type, resource)][0]
    #     print(f"  {assignment}: {processing_time}")
    
    assignment = min(possible_assignments, key=lambda x: simulator.problem.processing_time_distribution[(x[1], x[0])][0])
    resource, task_type = assignment
    
    # Print the selected resource-task type assignment
    #print(f"Selected assignment: {assignment} with processing time: {simulator.problem.processing_time_distribution[(task_type, resource)][0]}")

    # Find the corresponding task with the lowest task id in the ongoing tasks (Random + SPT policy)
    assignment = simulator.resource_to_task_type_assignment(resource, task_type)
    resource, task = assignment
    
    # Print the final resource-task assignment
    #print(f"Final assignment: ({resource}, {task})")
    
    return (resource, task)

def fifo_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None

    # Sort the list of tasks by their case_id attribute (assuming case_id represents arrival order)
    sorted_tasks = sorted(simulator.unassigned_tasks.values(), key=lambda task: task.case_id)
    
    # Iterate through tasks in FIFO order
    for task in sorted_tasks:
        # Find all resources that can process this task type
        matching_assignments = [(resource, task_type) for resource, task_type in possible_assignments 
                              if task.task_type == task_type]
        # print(task.case_id, matching_assignments)
        # print("eligbile resources:", [(resource, resource in simulator.available_resources) for resource in simulator.problem_resource_pool[task.task_type]])
        if matching_assignments:
            # Select the resource with the shortest processing time for this task type
            assignment = min(matching_assignments, key=lambda x: simulator.problem.processing_time_distribution[(x[1], x[0])][0])
            
            # Return the final assignment
            resource, _ = assignment
            #print("assignment made:", assignment)
            return (resource, task)
    return None

def random_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    # Randomly select a task and resource from the possible assignments
    resource, task_type = random.choice(possible_assignments)

    # Find the corresponding task with the lowest task id in the ongoing tasks (Random + SPT policy)
    assignment = simulator.resource_to_task_type_assignment(resource, task_type)
    resource, task = assignment
    return (resource, task)

def least_flexible_resource_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    # Only consider available resources as candidates
    candidate_resources = set(resource for resource, _ in possible_assignments)

    # Find the resource with the least flexibility (i.e., the fewest task types it can process)
    # For each resource, count how many different task types it can handle
    resource_flexibility = {}
    for resource in candidate_resources:
        resource_flexibility[resource] = sum(1 for task_type in simulator.task_types 
                                          if resource in simulator.problem_resource_pool[task_type])
    
    # Sort resources by flexibility (ascending)
    least_flexible_resources = sorted(candidate_resources, key=lambda r: resource_flexibility[r])

    # Try to assign tasks to resources in order of increasing flexibility
    for least_flexible_resource in least_flexible_resources:
        matching_assignments = [(resource, task_type) for resource, task_type in possible_assignments 
                               if resource == least_flexible_resource]
        if matching_assignments:
            # Select the assignment with the shortest processing time
            assignment = min(matching_assignments, 
                           key=lambda x: simulator.problem.processing_time_distribution[(x[1], x[0])][0])
            
            resource, task_type = assignment
            # Find the corresponding task
            assignment = simulator.resource_to_task_type_assignment(resource, task_type)
            resource, task = assignment
            return (resource, task)
    return None

def most_flexible_resource_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    # Only consider available resources as candidates
    candidate_resources = set(resource for resource, _ in possible_assignments)

    # Find the resource with the most flexibility (i.e., the most task types it can process)
    # For each resource, count how many different task types it can handle
    resource_flexibility = {}
    for resource in candidate_resources:
        resource_flexibility[resource] = sum(1 for task_type in simulator.task_types 
                                          if resource in simulator.problem_resource_pool[task_type])
    
    # Sort resources by flexibility (descending)
    most_flexible_resources = sorted(candidate_resources, key=lambda r: resource_flexibility[r], reverse=True)

    # Try to assign tasks to resources in order of decreasing flexibility
    for most_flexible_resource in most_flexible_resources:
        matching_assignments = [(resource, task_type) for resource, task_type in possible_assignments 
                               if resource == most_flexible_resource]
        if matching_assignments:
            # Select the assignment with the shortest processing time
            assignment = min(matching_assignments, 
                           key=lambda x: simulator.problem.processing_time_distribution[(x[1], x[0])][0])
            
            resource, task_type = assignment
            # Find the corresponding task
            assignment = simulator.resource_to_task_type_assignment(resource, task_type)
            resource, task = assignment
            return (resource, task)
    return None

def least_flexible_activity_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    # Only consider task types that have unassigned tasks
    candidate_task_types = set(task_type for _, task_type in possible_assignments)

    # Find the task type with the least flexibility (i.e., the fewest resources that can process it)
    # For each task type, determine how many resources can handle it
    task_flexibility = {}
    for task_type in candidate_task_types:
        task_flexibility[task_type] = len(simulator.get_resources_for_task_type(task_type))
    
    # Sort task types by flexibility (ascending)
    least_flexible_task_types = sorted(candidate_task_types, key=lambda t: task_flexibility[t])

    # Try to assign resources to task types in order of increasing flexibility
    for least_flexible_task_type in least_flexible_task_types:
        matching_assignments = [(resource, task_type) for resource, task_type in possible_assignments 
                               if task_type == least_flexible_task_type]
        if matching_assignments:
            # Select the assignment with the shortest processing time
            assignment = min(matching_assignments, 
                           key=lambda x: simulator.problem.processing_time_distribution[(x[1], x[0])][0])
            
            resource, task_type = assignment
            # Find the corresponding task
            assignment = simulator.resource_to_task_type_assignment(resource, task_type)
            resource, task = assignment
            return (resource, task)
    return None

def most_flexible_activity_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    # Only consider task types that have unassigned tasks
    candidate_task_types = set(task_type for _, task_type in possible_assignments)

    # Find the task type with the most flexibility (i.e., the most resources that can process it)
    # For each task type, determine how many resources can handle it
    task_flexibility = {}
    for task_type in candidate_task_types:
        task_flexibility[task_type] = len(simulator.get_resources_for_task_type(task_type))
    
    # Sort task types by flexibility (descending)
    most_flexible_task_types = sorted(candidate_task_types, key=lambda t: task_flexibility[t], reverse=True)

    # Try to assign resources to task types in order of decreasing flexibility
    for most_flexible_task_type in most_flexible_task_types:
        matching_assignments = [(resource, task_type) for resource, task_type in possible_assignments 
                               if task_type == most_flexible_task_type]
        if matching_assignments:
            # Select the assignment with the shortest processing time
            assignment = min(matching_assignments, 
                           key=lambda x: simulator.problem.processing_time_distribution[(x[1], x[0])][0])
            
            resource, task_type = assignment
            # Find the corresponding task
            assignment = simulator.resource_to_task_type_assignment(resource, task_type)
            resource, task = assignment
            return (resource, task)
    return None

def hrrn_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    best_assignment = None
    highest_hrrn = -1

    # For each possible assignment, calculate HRRN and select the highest
    for resource, task_type in possible_assignments:
        # Find the corresponding task
        assignment = simulator.resource_to_task_type_assignment(resource, task_type)
        if assignment[0] is None:  # If no task available for this type
            continue
            
        resource, task = assignment
        
        # Calculate expected processing time for this resource-task combination
        expected_pt = simulator.problem.processing_time_distribution[(task.task_type, resource)][0]
        
        # Calculate waiting time for this task
        waiting_time = simulator.now - simulator.task_arrival_times[task.id]
        
        # Calculate HRRN
        hrrn_score = (waiting_time + expected_pt) / expected_pt
        
        # Update best assignment if this one has higher HRRN
        if hrrn_score > highest_hrrn:
            highest_hrrn = hrrn_score
            best_assignment = (resource, task)
    
    return best_assignment

def shortest_queue_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
      
    queue_lengths = simulator.get_queue_lengths_per_task_type()

    # Sort the task types by their queue lengths (ascending)
    sorted_task_types = sorted(queue_lengths.items(), key=lambda x: x[1])

    # Iterate through task types with the shortest queue first
    for task_type, queue_length in sorted_task_types:
        if queue_length > 0:
            # Find all possible assignments for this task type
            matching_assignments = [(resource, t_type) for resource, t_type in possible_assignments 
                                   if t_type == task_type]
            if matching_assignments:
                # Select the assignment with the shortest processing time
                assignment = min(matching_assignments, 
                               key=lambda x: simulator.problem.processing_time_distribution[(x[1], x[0])][0])
                
                resource, task_type = assignment
                # Find the corresponding task
                assignment = simulator.resource_to_task_type_assignment(resource, task_type)
                resource, task = assignment
                return (resource, task)
    return None

def longest_queue_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
      
    queue_lengths = simulator.get_queue_lengths_per_task_type()

    # Sort the task types by their queue lengths (descending)
    sorted_task_types = sorted(queue_lengths.items(), key=lambda x: x[1], reverse=True)

    # Iterate through task types with the longest queue first
    for task_type, queue_length in sorted_task_types:
        if queue_length > 0:
            # Find all possible assignments for this task type
            matching_assignments = [(resource, t_type) for resource, t_type in possible_assignments 
                                   if t_type == task_type]
            if matching_assignments:
                # Select the assignment with the shortest processing time
                assignment = min(matching_assignments, 
                               key=lambda x: simulator.problem.processing_time_distribution[(x[1], x[0])][0])
                
                resource, task_type = assignment
                # Find the corresponding task
                assignment = simulator.resource_to_task_type_assignment(resource, task_type)
                resource, task = assignment
                return (resource, task)
    return None