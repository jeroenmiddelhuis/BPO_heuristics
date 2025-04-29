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
            for resource, task_type in possible_assignments:
                if task.task_type == task_type:
                    return (resource, task)
    return None
