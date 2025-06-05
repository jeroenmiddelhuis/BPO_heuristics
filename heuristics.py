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
    min_pt = min(x[0].processing_times[x[1]] for x in possible_assignments)
    min_assignments = [x for x in possible_assignments if x[0].processing_times[x[1]] == min_pt]
    assignment = random.choice(min_assignments)
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
                # Select all resources with the shortest processing time for this task type
                min_pt = min(x[0].processing_times[x[1]] for x in matching_resources)
                best_resources = [x for x in matching_resources if x[0].processing_times[x[1]] == min_pt]
                best_resource, task_type = random.choice(best_resources)
                
                return (best_resource, task)
    
    return None

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
            matching_assignments = [(resource, t_type) for resource, t_type in possible_assignments if t_type == task_type]
            if matching_assignments:
                # Select the assignment with the shortest processing time
                min_pt = min(x[0].processing_times[x[1]] for x in matching_assignments)
                best_assignments = [x for x in matching_assignments if x[0].processing_times[x[1]] == min_pt]
                best_resource, best_task_type = random.choice(best_assignments)
                # Find the corresponding task
                assignment = simulator.resource_to_task_type_assignment(best_resource, best_task_type)
                resource, task = assignment
                return (resource, task)
    return None


def longest_queue_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
      
    queue_lengths = simulator.get_queue_lengths_per_task_type()

    # Sort the task types by their queue lengths (ascending)
    sorted_task_types = sorted(queue_lengths.items(), key=lambda x: x[1], reverse=True)

    # Iterate through task types with the shortest queue first
    for task_type, queue_length in sorted_task_types:
        if queue_length > 0:
            # Find all possible assignments for this task type
            matching_assignments = [(resource, t_type) for resource, t_type in possible_assignments if t_type == task_type]
            if matching_assignments:
                # Select the assignment with the shortest processing time
                min_pt = min(x[0].processing_times[x[1]] for x in matching_assignments)
                best_assignments = [x for x in matching_assignments if x[0].processing_times[x[1]] == min_pt]
                best_resource, best_task_type = random.choice(best_assignments)
                # Find the corresponding task
                assignment = simulator.resource_to_task_type_assignment(best_resource, best_task_type)
                resource, task = assignment
                return (resource, task)
    return None

def hrrn_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    best_assignment = None
    hrrn = -1

    for case in simulator.ongoing_cases:
        random.shuffle(case.ongoing_tasks)
        for task in case.ongoing_tasks:
            # Find all resources that can process this task type
            matching_assignments = [(resource, task_type) for resource, task_type in possible_assignments 
                                 if task.task_type == task_type]
            
            if matching_assignments:
                for resource, task_type in matching_assignments:
                    expected_pt = resource.processing_times[task_type]
                    waiting_time = simulator.now - task.arrival_time
                    hrrn_score = (waiting_time + expected_pt) / expected_pt
                    if hrrn_score > hrrn:
                        hrrn = hrrn_score
                        best_assignment = (resource, task)
                
    return best_assignment

def least_flexible_resource_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    # Only consider available resources as candidates
    candidate_resources = set(resource for resource, _ in possible_assignments if resource.available)

    # Find the resource with the least flexibility (i.e., the fewest task types it can process)
    least_flexible_resources = sorted(candidate_resources, key=lambda r: len(r.processing_times))

    for least_flexible_resource in least_flexible_resources:
        matching_assignments = [(resource, task_type) for resource, task_type in possible_assignments if resource == least_flexible_resource]
        if matching_assignments:
            # Select the assignment with the shortest processing time
            min_pt = min(x[0].processing_times[x[1]] for x in matching_assignments)
            best_assignments = [x for x in matching_assignments if x[0].processing_times[x[1]] == min_pt]
            best_resource, best_task_type = random.choice(best_assignments)
            # Find the corresponding task
            assignment = simulator.resource_to_task_type_assignment(best_resource, best_task_type)
            resource, task = assignment
            return (resource, task)
    return None

def most_flexible_resource_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    # Only consider available resources as candidates
    candidate_resources = set(resource for resource, _ in possible_assignments if resource.available)

    # Find the resource with the most flexibility (i.e., the fewest task types it can process)
    most_flexible_resources = sorted(candidate_resources, key=lambda r: len(r.processing_times), reverse=True)

    for most_flexible_resource in most_flexible_resources:
        matching_assignments = [(resource, task_type) for resource, task_type in possible_assignments if resource == most_flexible_resource]
        if matching_assignments:
            # Select the assignment with the shortest processing time
            min_pt = min(x[0].processing_times[x[1]] for x in matching_assignments)
            best_assignments = [x for x in matching_assignments if x[0].processing_times[x[1]] == min_pt]
            best_resource, best_task_type = random.choice(best_assignments)
            # Find the corresponding task
            assignment = simulator.resource_to_task_type_assignment(best_resource, best_task_type)
            resource, task = assignment
            return (resource, task)
    return None

def least_flexible_activity_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    # Only consider available resources as candidates
    candidate_task_types = set(task_type for _, task_type in possible_assignments)

    # Find the task type with the least flexiblity (i.e., the least resources that can process it)
    least_flexible_task_types = sorted(candidate_task_types, key=lambda t: len(simulator.get_resources_for_task_type(t)))

    for least_flexible_task_type in least_flexible_task_types:
        matching_assignments = [(resource, task_type) for resource, task_type in possible_assignments if task_type == least_flexible_task_type]
        if matching_assignments:
            # Select the assignment with the shortest processing time
            min_pt = min(x[0].processing_times[x[1]] for x in matching_assignments)
            best_assignments = [x for x in matching_assignments if x[0].processing_times[x[1]] == min_pt]
            best_resource, best_task_type = random.choice(best_assignments)
            # Find the corresponding task
            assignment = simulator.resource_to_task_type_assignment(best_resource, best_task_type)
            resource, task = assignment
            return (resource, task)
    return None

def most_flexible_activity_policy(simulator):
    possible_assignments = simulator.get_possible_resource_to_task_type_assignments()
    if not possible_assignments:
        return None
    
    # Only consider available resources as candidates
    candidate_task_types = set(task_type for _, task_type in possible_assignments)

    # Find the task type with the most flexiblity (i.e., the most resources that can process it)
    most_flexible_task_types = sorted(candidate_task_types, key=lambda t: len(simulator.get_resources_for_task_type(t)), reverse=True)

    for most_flexible_task_type in most_flexible_task_types:
        matching_assignments = [(resource, task_type) for resource, task_type in possible_assignments if task_type == most_flexible_task_type]
        if matching_assignments:
            # Select the assignment with the shortest processing time
            min_pt = min(x[0].processing_times[x[1]] for x in matching_assignments)
            best_assignments = [x for x in matching_assignments if x[0].processing_times[x[1]] == min_pt]
            best_resource, best_task_type = random.choice(best_assignments)
            # Find the corresponding task
            assignment = simulator.resource_to_task_type_assignment(best_resource, best_task_type)
            resource, task = assignment
            return (resource, task)
    return None