import random

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

def spt_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    # From the possible resource to task_type assignments, select the one with the lowest expected processing time
    assignment = min(possible_assignments, key=lambda x: x[0].processing_times[x[1]])
    resource, task_type = assignment

    # Find the corresponding task with the lowest task id in the ongoing tasks (Random + SPT policy)
    assignment = simulator.resource_to_task_type_assignment(resource, task_type)
    resource, task = assignment
    return (resource, task)

def fifo_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None

    # Select the assignment with the lowest case id
    for case in simulator.ongoing_cases:
        random.shuffle(case.ongoing_tasks)  # Shuffle the ongoing tasks to randomize the selection
        for task in case.ongoing_tasks:
            # Find all resources that can process this task type
            matching_resources = [(resource, task_type) for resource, task_type in possible_assignments 
                                 if task.task_type == task_type]
            
            if matching_resources:
                # Select the resource with the shortest processing time for this task type
                best_resource, task_type = min(matching_resources, 
                                              key=lambda x: x[0].processing_times[x[1]])
                return (best_resource, task)
    
    return None

def shortest_queue_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    # Calculate workload for each resource based on currently assigned tasks
    resource_workloads = {}
    for resource in simulator.resources:
        # A resource has workload 1 if it's busy (not available), 0 if available
        resource_workloads[resource] = 0 if resource.is_available() else 1
    
    # Group resources by their workload
    resources_by_workload = {}
    for resource, task_type in possible_assignments:
        workload = resource_workloads[resource]
        if workload not in resources_by_workload:
            resources_by_workload[workload] = []
        resources_by_workload[workload].append((resource, task_type))
    
    # Find the least loaded resources (ones with minimum workload)
    min_workload = min(resources_by_workload.keys())
    
    # Among resources with minimum workload, select the one with shortest processing time
    candidates = resources_by_workload[min_workload]
    best_assignment = min(candidates, key=lambda x: x[0].processing_times[x[1]])
    resource, task_type = best_assignment
    
    # Find the corresponding task
    assignment = simulator.resource_to_task_type_assignment(resource, task_type)
    resource, task = assignment
    return (resource, task)

def longest_queue_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    # Calculate workload for each resource based on currently assigned tasks
    resource_workloads = {}
    for resource in simulator.resources:
        # A resource has workload 1 if it's busy (not available), 0 if available
        resource_workloads[resource] = 0 if resource.is_available() else 1
    
    # Group resources by their workload
    resources_by_workload = {}
    for resource, task_type in possible_assignments:
        workload = resource_workloads[resource]
        if workload not in resources_by_workload:
            resources_by_workload[workload] = []
        resources_by_workload[workload].append((resource, task_type))
    
    # Find the most loaded resources (ones with maximum workload)
    max_workload = max(resources_by_workload.keys())
    
    # Among resources with maximum workload, select the one with shortest processing time
    candidates = resources_by_workload[max_workload]
    best_assignment = min(candidates, key=lambda x: x[0].processing_times[x[1]])
    resource, task_type = best_assignment
    
    # Find the corresponding task
    assignment = simulator.resource_to_task_type_assignment(resource, task_type)
    resource, task = assignment
    return (resource, task)