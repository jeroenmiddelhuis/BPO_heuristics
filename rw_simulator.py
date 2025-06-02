import json
from enum import Enum, auto
from datetime import datetime, timedelta
import numpy as np
import random
import pickle as pickle
from abc import ABC, abstractmethod
import numpy as np
# from types import NoneType

from time import time

from matplotlib import pyplot as plt


class Event:
    initial_time = datetime(2020, 1, 1)
    time_format = "%Y-%m-%d %H:%M:%S.%f"

    def __init__(self, case_id, task, timestamp, resource, lifecycle_state):
        self.case_id = case_id
        self.task = task
        self.timestamp = timestamp
        self.resource = resource
        self.lifecycle_state = lifecycle_state

    def __str__(self):
        t = (self.initial_time + timedelta(hours=self.timestamp)).strftime(self.time_format)
        return str(self.case_id) + "\t" + str(self.task) + "\t" + t + "\t" + str(self.resource) + "\t" + str(
            self.lifecycle_state)


class Task:

    def __init__(self, task_id, case_id, task_type):
        self.id = task_id
        self.case_id = case_id
        self.task_type = task_type

    def __lt__(self, other):
        return self.id < other.id

    def __str__(self):
        return self.task_type


class Problem(ABC):

    @property
    @abstractmethod
    def resources(self):
        raise NotImplementedError

    @property
    def resource_weights(self):
        return self._resource_weights

    @resource_weights.setter
    def resource_weights(self, value):
        self._resource_weights = value

    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self, value):
        self._schedule = value

    @property
    @abstractmethod
    def task_types(self):
        raise NotImplementedError

    @abstractmethod
    def sample_initial_task_type(self):
        raise NotImplementedError

    def resource_pool(self, task_type):
        return self.resources

    def __init__(self, deterministic_processing=True):
        self.next_case_id = 0
        self.cases = dict()  # case_id -> (arrival_time, initial_task)
        self._resource_weights = [1] * len(self.resources)
        self._schedule = [len(self.resources)]
        self._task_processing_times = dict()
        self._task_next_tasks = dict()
        self.deterministic_processing = deterministic_processing

    def from_generator(self, duration):
        now = 0
        next_case_id = 0
        next_task_id = 0
        unfinished_tasks = []
        # Instantiate cases at the interarrival time for the duration.
        # Generate the first task for each case, without processing times and next tasks, add them to the unfinished tasks.
        while now < duration:
            at = now + self.interarrival_time_sample()
            initial_task_type = self.sample_initial_task_type()
            task = Task(next_task_id, next_case_id, initial_task_type)
            next_task_id += 1
            unfinished_tasks.append(task)
            self.cases[next_case_id] = (at, task)
            next_case_id += 1
            now = at
        # Finish the tasks by:
        # 1. generating the processing times.
        # 2. generating the next tasks, without processing times and next tasks, add them to the unfinished tasks.
        while len(unfinished_tasks) > 0:
            task = unfinished_tasks.pop(0)
            for r in self.resource_pool(task.task_type):
                pt = self.processing_time_sample(r, task, self.deterministic_processing)
                if task not in self._task_processing_times:
                    self._task_processing_times[task] = dict()
                self._task_processing_times[task][r] = pt
            for tt in self.next_task_types_sample(task):
                new_task = Task(next_task_id, task.case_id, tt)
                next_task_id += 1
                unfinished_tasks.append(new_task)
                if task not in self._task_next_tasks:
                    self._task_next_tasks[task] = []
                self._task_next_tasks[task].append(new_task)
        return self

    @classmethod
    def from_file(cls, filename):
        # import pdb; pdb.set_trace()
        with open(filename, 'rb') as handle:
            instance = pickle.load(handle)
        return instance

    def save_instance(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def processing_time_sample(self, resource, task, deterministic_time):
        raise NotImplementedError

    @abstractmethod
    def interarrival_time_sample(self):
        raise NotImplementedError

    def next_task_types_sample(self, task):
        return []

    def restart(self):
        self.next_case_id = 0

    def next_case(self):
        try:
            (arrival_time, initial_task) = self.cases[self.next_case_id]
            self.next_case_id += 1
            return arrival_time, initial_task
        except KeyError:
            return None

    def next_tasks(self, task):
        if task in self._task_next_tasks:
            return self._task_next_tasks[task]
        else:
            return []

    def processing_time(self, task, resource):
        return self._task_processing_times[task][resource]


class MinedProblem(Problem):
    resources = []
    task_types = []

    def __init__(self):
        super().__init__()
        self.initial_task_distribution = []
        self.next_task_distribution = dict()
        self.mean_interarrival_time = 0
        self.resource_pools = dict()
        self.processing_time_distribution = dict()


    def sample_initial_task_type(self):
        rd = random.random()
        rs = 0
        for (p, tt) in self.initial_task_distribution:
            rs += p
            if rd < rs:
                return tt
        print("WARNING: the probabilities of initial tasks do not add up to 1.0")
        return self.initial_task_distribution[0]

    def resource_pool(self, task_type):
        return self.resource_pools[task_type]

    def interarrival_time_sample(self):
        return random.expovariate(1 / (self.mean_interarrival_time))

    def next_task_types_sample(self, task):
        rd = random.random()
        rs = 0
        for (p, tt) in self.next_task_distribution[task.task_type]:
            rs += p
            if rd < rs:
                if tt is None:
                    return []
                else:
                    return [tt]
        print("WARNING: the probabilities of next tasks do not add up to 1.0")
        if self.next_task_distribution[0][1] is None:
            return []
        else:
            return [self.next_task_distribution[0][1]]

    def processing_time_sample(self, resource, task, deterministic_processing):
        # Gamma
        (mu, sigma) = self.processing_time_distribution[(task.task_type, resource)]
        # alpha = mu**2/sigma**2
        # beta = mu/sigma**2
        if deterministic_processing:
            return mu

        # We do not allow negative values for processing time.
        pt = random.gauss(mu, np.sqrt(sigma))
        while pt < 0:
            pt = random.gauss(mu, np.sqrt(sigma))
        return pt

    @classmethod
    def generator_from_file(cls, filename):
        o = MinedProblem()
        with open(filename, 'rb') as handle:
            o.resources = pickle.load(handle)
            o.task_types = pickle.load(handle)
            o.initial_task_distribution = pickle.load(handle)
            o.next_task_distribution = pickle.load(handle)
            o.mean_interarrival_time = pickle.load(handle)
            o.resource_pools = pickle.load(handle)
            o.processing_time_distribution = pickle.load(handle)
            o.resource_weights = pickle.load(handle)
            o.schedule = pickle.load(handle)
        return o

    def save_generator(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.resources, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.task_types, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.initial_task_distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.next_task_distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.mean_interarrival_time, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.resource_pools, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.processing_time_distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.resource_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.schedule, handle, protocol=pickle.HIGHEST_PROTOCOL)


class EventType(Enum):
    CASE_ARRIVAL = auto()
    START_TASK = auto()
    COMPLETE_TASK = auto()
    PLAN_TASKS = auto()
    TASK_ACTIVATE = auto()
    TASK_PLANNED = auto()
    COMPLETE_CASE = auto()
    SCHEDULE_RESOURCES = auto()
    RETURN_REWARD = auto()


class TimeUnit(Enum):
    SECONDS = auto()
    MINUTES = auto()
    HOURS = auto()
    DAYS = auto()


class SimulationEvent:
    def __init__(self, event_type, moment, task, resource=None, nr_tasks=0, nr_resources=0):
        self.event_type = event_type
        self.moment = moment
        self.task = task
        self.resource = resource
        self.nr_tasks = nr_tasks
        self.nr_resources = nr_resources

    def __lt__(self, other):
        return self.moment < other.moment

    def __str__(self):
        return str(self.event_type) + "\t(" + str(round(self.moment, 2)) + ")\t" + str(self.task) + "," + str(
            self.resource)


class Simulator:
    def __init__(self, 
                 nr_cases, 
                 report=False, 
                 problem=None, 
                 instance_file="BPI Challenge 2017 - instance.pickle",
                 planner=None, 
                 record_total_cases=False, 
                 normalize_nodes_attrs=False, 
                 max_tasks=0, 
                 interarrival_rate_multiplier=1, 
                 record_states=False, 
                 max_transitions=0, 
                 deterministic_processing=False):

        self.report = report
        self.events = []
        self.planner = planner
        self.multi_agent = False
        self.deterministic_processing = deterministic_processing

        #flags to record problem characteristics
        self.record_total_cases = record_total_cases
        self.record_states = record_states
        self.total_cases_dict = {'time': [], 'total_cases': []}

        #self.instance_file is instance_file without extention
        self.instance_file = instance_file.split('/')[-1].split('.')[0]

        self.unassigned_tasks = dict()
        self.assigned_tasks = dict()
        self.available_resources = set()
        self.away_resources = []
        self.away_resources_weights = []
        self.just_gone_away_resources = []
        self.just_finished_task_resources = []
        self.busy_resources = dict()
        self.busy_cases = dict()
        self.reserved_resources = dict()
        self.now = 0
        self.last_decision_time = 0

        self.n_finalized_cases = 0
        self.finalized_cases = []
        self.total_cycle_time = 0
        self.case_start_times = dict()
        self.task_start_times = dict()
        self.task_arrival_times = dict()

        self.observation_time = 0
        self.observation_time_old = 0

        if problem is None:
            # self.problem = MinedProblem.from_file(instance_file)
            self.problem = MinedProblem.from_file(instance_file)
            self.problem.mean_interarrival_time = self.problem.mean_interarrival_time


        elif isinstance(problem, MinedProblem):
            self.problem = problem
            self.problem.mean_interarrival_time = self.problem.mean_interarrival_time / interarrival_rate_multiplier
        else:
            raise Exception("Problem is not correctly initialized")

        self.problem_resource_pool = self.problem.resource_pools

        # parameters for task generationF
        self.next_case_id = 0
        self.next_task_id = 0

        # new parameters
        self.nr_cases = nr_cases
        self.max_tasks = max_tasks
        self.status = "RUNNING"
        self.count_rewards = 0
        self.return_reward = False
        self.plan = False
        self.average_cycle_time = None
        self.previous_average_cycle_time = None  # used to split the rewards daily
        self.previous_finalized_cases = 0
        self.resources_scheduled = False
        self.last_assignment_time = 0  # used to track the moment of the last assignment
        self.last_assignment_duration = 0  # used to track the duration since the last assignment
        self.total_reward = 0
        self.case_completed_tasks = dict()  # ADDED

        self.total_completed_tasks = 0
        self.residual_cycle_time = 0

        self.reward_task = 0
        self.reward_case = 0
        self.reward_penalty = 0
        self.temp = 1

        self.max_transitions = max_transitions
        self.transitions_num = 0

        # make sure the task types and resources are encoded as strings
        self.problem_resource_pool = {str(key): [str(resource) for resource in value] for key, value in
                                      self.problem_resource_pool.items()}

        # parameters needed for masking
        self.task_types = sorted(list(self.problem_resource_pool.keys()))  # all task types (should be 7 elements)

        self.resources = sorted(list(set(np.hstack(
            list(self.problem_resource_pool.values())))))  # all the resources in the problem (should be 145 elements)

        self.unassigned_tasks_per_type = {task_type: [] for task_type in self.task_types}

        self.resource_assignability = np.zeros((len(self.task_types), len(self.resources)))
        for idx_t, task_type in enumerate(self.task_types):
            for idx_r, resource in enumerate(self.resources):
                ass_list = []
                if resource in self.problem_resource_pool[task_type]:
                    ass_list.append(idx_r)
                self.resource_assignability[idx_t][ass_list] = 1
        self.resource_assignability = self.resource_assignability.T
        print(self.resource_assignability)

        # save rewards for each agent
        #self.current_rewards = {key: 0 for key in self.resources}
        self.reward = 0
        #self.case_rewards = {key: 0 for key in self.resources}
        self.last_event_time = 0

        self.debug_report = False

        self.output = [(resource, task) for resource in self.resources for task in self.task_types if
                       task in self.problem.resource_pools and resource in self.problem.resource_pools[task]]

        # edge feature: average completion time for (resource,task_type)
        self.edge_features = np.zeros(len(self.output))
        # the average completion time for each resource-task pair is in self.problem.processing_time_distribution
        for idx, resource_task_type in enumerate(self.output):  # for now, postpone is not considered
            self.edge_features[idx] = \
            self.problem.processing_time_distribution[resource_task_type[1], resource_task_type[0]][0]

        # used for getting graph state representation efficiently
        # Pre-calculate resource and task type indices
        self.resources_set = set(self.resources)
        self.task_types_set = set(self.task_types)
        self.resource_pools_sets = {task_type: set(self.problem.resource_pools[task_type]) for task_type in
                                    self.task_types}
        self.resource_indices = {resource: i for i, resource in enumerate(self.resources)}
        self.task_type_indices = {task_type: i for i, task_type in enumerate(self.task_types)}

        # make sure the order is always the same
        self.output.sort()
        self.output = self.output + ['Postpone']
        self.init_simulation()

        self.edge_index = [[self.resource_indices[resource], self.task_type_indices[task_type]]
                           for resource in self.resources for task_type in self.task_types
                           if resource in self.problem.resource_pools[task_type]]

        self.assignment_nodes = []
        self.assignment_nodes_attr = []
        for resource_index, task_type_index in self.edge_index:
            assignment_node = {'resource': resource_index, 'task_type': task_type_index}
            self.assignment_nodes_attr.append(self.edge_features[self.output.index(
                (self.resources[resource_index], self.task_types[task_type_index]))])
            self.assignment_nodes.append(assignment_node)

        if normalize_nodes_attrs:
            self.assignment_nodes_attr = self.normalize_features(torch.tensor(self.assignment_nodes_attr)).numpy()

    def init_simulation(self):
        # set all resources to available
        for r in self.problem.resources:
            self.available_resources.add(r)

        # generate resource scheduling event to start the schedule
        self.events.append((self.now, SimulationEvent(EventType.SCHEDULE_RESOURCES, self.now, None)))

        # reset the problem
        self.problem.restart()

        # generate arrival event for the first task of the first case
        (t, task) = self.generate_case()
        self.events.append((t, SimulationEvent(EventType.CASE_ARRIVAL, t, task)))

    def desired_nr_resources(self):
        return self.problem.schedule[int(self.now % len(self.problem.schedule))]

    def working_nr_resources(self):
        return len(self.available_resources) + len(self.busy_resources) + len(self.reserved_resources)

    def generate_case(self):
        t = self.now + self.problem.interarrival_time_sample()
        initial_task_type = self.problem.sample_initial_task_type()
        task = Task(self.next_task_id, self.next_case_id, initial_task_type)
        self.case_completed_tasks[task.case_id] = 0  # ADDED
        self.next_task_id += 1
        self.next_case_id += 1
        return (t, task)

    def generate_next_tasks(self, task):
        for tt in self.problem.next_task_types_sample(task):
            new_task = Task(self.next_task_id, task.case_id, tt)
            self.unassigned_tasks[new_task.id] = new_task
            self.unassigned_tasks_per_type[new_task.task_type].append(new_task.id)
            self.task_arrival_times[new_task.id] = self.now
            self.busy_cases[task.case_id].append(new_task.id)
            self.next_task_id += 1

    def run_until_next_decision_epoch(self):
        # reward is reset after every action
        self.total_reward += self.reward
        self.reward = 0

        # repeat until the end of the simulation time:
        while not self.is_done():
            # get the first event e from the events
            self.events.sort()
            event = self.events.pop(0)
            # t = time of e
            self.previous_time = self.now
            self.now = event[0]

            self.update_rewards()
            event = event[1]
            # print(f"Event {event.event_type} at time: {self.now} leading to reward: {self.current_reward}")

            if self.record_total_cases:
                self.total_cases_dict['time'].append(self.now)
                self.total_cases_dict['total_cases'].append(len(self.assigned_tasks) + len(self.unassigned_tasks))

            # if e is an arrival event:
            if event.event_type == EventType.CASE_ARRIVAL:

                self.case_start_times[event.task.case_id] = self.now
                if self.report: print(Event(event.task.case_id, None, self.now, None, EventType.CASE_ARRIVAL))
                # add new task
                if self.report: print(Event(event.task.case_id, event.task, self.now, None, EventType.TASK_ACTIVATE))
                self.unassigned_tasks[event.task.id] = event.task
                self.unassigned_tasks_per_type[event.task.task_type].append(event.task.id)
                self.task_arrival_times[event.task.id] = self.now
                self.busy_cases[event.task.case_id] = [event.task.id]
                # generate a new planning event to start planning now for the new task
                self.events.append((self.now, SimulationEvent(EventType.PLAN_TASKS, self.now, None,
                                                              nr_tasks=len(self.unassigned_tasks),
                                                              nr_resources=len(self.available_resources))))
                # generate a new arrival event for the first task of the next case
                if self.next_case_id < self.nr_cases:
                    (t, task) = self.generate_case()
                    self.events.append((t, SimulationEvent(EventType.CASE_ARRIVAL, t, task)))
                # self.events.sort()


            # if e is a start event:
            elif event.event_type == EventType.START_TASK:
                if self.report: print(
                    Event(event.task.case_id, event.task, self.now, event.resource, EventType.START_TASK))
                if self.debug_report:
                    print(f"Task {event.task} assigned to resource {event.resource} at time {self.now}")
                # create a complete event for task
                t = self.now + self.problem.processing_time_sample(event.resource, event.task, self.deterministic_processing)
                self.events.append((t, SimulationEvent(EventType.COMPLETE_TASK, t, event.task, event.resource)))
                # self.events.sort()
                # set resource to busy
                del self.reserved_resources[event.resource]
                self.busy_resources[event.resource] = (event.task, self.now)

                # create an entry in self.task_start_times to keep track of the task's cycle times
                self.task_start_times[event.task.id] = self.now
                self.last_decision_time = self.now

            # if e is a complete event:
            elif event.event_type == EventType.COMPLETE_TASK:
                if self.debug_report:
                    print(
                        f"Task {event.task.id} finished by resource {event.resource}. Started at {self.task_start_times[event.task.id]}, arrived at {self.task_arrival_times[event.task.id]}, finished at {self.now}. Expected reward {self.now - self.task_arrival_times[event.task.id]}")
                if self.report: print(
                    Event(event.task.case_id, event.task, self.now, event.resource, EventType.COMPLETE_TASK))
                # set resource to available, if it is still desired, otherwise set it to away

                self.busy_cases[event.task.case_id].remove(event.task.id)
                self.case_completed_tasks[event.task.case_id] += 1  # ADDED

                self.generate_next_tasks(event.task)

                if len(self.busy_cases[event.task.case_id]) == 0:
                    self.case_rewards[event.resource] += self.now - self.case_start_times[event.task.case_id]
                    if self.report: self.planner.report(
                        Event(event.task.case_id, None, self.now, None, EventType.COMPLETE_CASE))
                    self.events.append((self.now, SimulationEvent(EventType.COMPLETE_CASE, self.now, event.task)))

                # remove task from assigned tasks
                del self.assigned_tasks[event.task.id]

                del self.busy_resources[event.resource]
                if self.working_nr_resources() <= self.desired_nr_resources():
                    self.available_resources.add(event.resource)
                    self.just_finished_task_resources.append(event.resource)
                else:
                    self.just_gone_away_resources.append(event.resource)
                    if self.debug_report:
                        print(f"Resource {event.resource} left the system after completing a task")
                    self.away_resources.append(event.resource)
                    self.away_resources_weights.append(
                        self.problem.resource_weights[self.problem.resources.index(event.resource)])

                # generate a new planning event to start planning now for the newly available resource and next tasks
                self.events.append((self.now, SimulationEvent(EventType.PLAN_TASKS, self.now, None,
                                                              nr_tasks=len(self.unassigned_tasks),
                                                              nr_resources=len(self.available_resources))))
                # self.events.sort()

                self.task_arrival_times.pop(event.task.id)
                self.task_start_times.pop(event.task.id)

                self.total_completed_tasks += 1

                # self.current_reward += self.reward_task

            # if e is a schedule resources event: move resources between available/away,
            # depending on how many resources should be available according to the schedule.
            elif event.event_type == EventType.SCHEDULE_RESOURCES:

                assert self.working_nr_resources() + len(self.away_resources) == len(
                    self.problem.resources)  # the number of resources must be constant
                assert len(self.problem.resources) == len(
                    self.problem.resource_weights)  # each resource must have a resource weight
                assert len(self.away_resources) == len(
                    self.away_resources_weights)  # each away resource must have a resource weight
                if len(self.away_resources) > 0:  # for each away resource, the resource weight must be taken from the problem resource weights
                    i = random.randrange(len(self.away_resources))
                    assert self.away_resources_weights[i] == self.problem.resource_weights[
                        self.problem.resources.index(self.away_resources[i])]
                required_resources = self.desired_nr_resources() - self.working_nr_resources()
                if required_resources > 0:
                    # if there are not enough resources working
                    # randomly select away resources to work, as many as required
                    for i in range(required_resources):
                        random_resource = random.choices(self.away_resources, self.away_resources_weights)[0]
                        # remove them from away and add them to available resources
                        away_resource_i = self.away_resources.index(random_resource)
                        del self.away_resources[away_resource_i]
                        del self.away_resources_weights[away_resource_i]
                        self.available_resources.add(random_resource)
                    # generate a new planning event to put them to work
                    self.events.append((self.now, SimulationEvent(EventType.PLAN_TASKS, self.now, None,
                                                                  nr_tasks=len(self.unassigned_tasks),
                                                                  nr_resources=len(self.available_resources))))
                    # self.events.sort()
                elif required_resources < 0:
                    # if there are too many resources working
                    # remove as many as possible, i.e. min(available_resources, -required_resources)
                    nr_resources_to_remove = min(len(self.available_resources), -required_resources)
                    resources_to_remove = random.sample(list(self.available_resources), nr_resources_to_remove)
                    for r in resources_to_remove:
                        # remove them from the available resources
                        self.available_resources.remove(r)
                        # add them to the away resources
                        self.away_resources.append(r)
                        self.just_gone_away_resources.append(r)
                        self.away_resources_weights.append(
                            self.problem.resource_weights[self.problem.resources.index(r)])
                # plan the next resource schedule event
                self.events.append((self.now + 1, SimulationEvent(EventType.SCHEDULE_RESOURCES, self.now + 1, None)))

            # if e is a planning event: do assignment
            elif event.event_type == EventType.PLAN_TASKS:
                return True

            # if e is a complete case event: add to the number of completed cases
            elif event.event_type == EventType.COMPLETE_CASE:

                self.total_cycle_time += self.now - self.case_start_times[event.task.case_id]
                self.n_finalized_cases += 1
                self.finalized_cases.append(event.task.case_id)

        if self.is_done():
            self.status = "FINISHED"

        if self.status == "FINISHED":
            #self.update_rewards()
            self.total_reward += self.reward

            if self.n_finalized_cases:
                print(
                    f"COMPLETED: you completed a full year of simulated customer cases. Average cycle time was {self.total_cycle_time / self.n_finalized_cases}")
                print(f"Time as a function of reward: {self.total_reward / self.n_finalized_cases}")
                return self.total_cycle_time / self.n_finalized_cases
            else:
                print(f"COMPLETED: you completed a full year of simulated customer cases. No cases completed.")
                return 0

    def schedule_resources(self, assignments):

        assignments_list = [
            (resource, next((x for x in list(self.unassigned_tasks.values()) if x.task_type == task), None)) for
            (resource, task) in assignments]

        # for each newly assigned task:
        moment = self.now
        self.last_assignment_duration = moment - self.last_assignment_time
        self.last_assignment_time = moment

        # for each newly assigned task:
        moment = self.now
        for el in assignments_list:
            task = el[1]
            resource = el[0]

            # print('EL:', el)
            if task.id not in [t.id for t in self.unassigned_tasks.values()]:
                return None, "ERROR: trying to assign a task that is not in the unassigned_tasks."
            if resource not in self.available_resources:
                return None, "ERROR: trying to assign a resource that is not in available_resources."
            if resource not in self.problem_resource_pool[task.task_type]:
                return None, "ERROR: trying to assign a resource to a task that is not in its resource pool."
            # create start event for task
            self.events.append((moment, SimulationEvent(EventType.START_TASK, moment, task, resource)))
            # assign task
            del self.unassigned_tasks[task.id]
            self.unassigned_tasks_per_type[task.task_type].remove(task.id)
            self.assigned_tasks[task.id] = (task, resource, moment)
            # reserve resource
            self.available_resources.remove(resource)
            self.reserved_resources[resource] = (task, moment)
        # self.events.sort()

        # Return assigned task id to get add waiting time to the cumulative reward when the action is taken
        if self.multi_agent:
            return assignments_list[0][0].id
        else:

            return assignments_list[0]

    def is_done(self):
        return self.n_finalized_cases == self.nr_cases

    def plot_cases(self):
        # Plot the number of cases in the system over time
        plt.plot(self.total_cases_dict['time'], self.total_cases_dict['total_cases'])
        plt.xlabel('Time')
        plt.ylabel('Number of cases in the system')
        plt.title('Number of cases in the system over time')
        plt.show()

        #Save the numbers to a txt file
        with open('total_cases.txt', 'w') as f:
            for i in range(len(self.total_cases_dict['time'])):
                f.write(f"{self.total_cases_dict['time'][i]} {self.total_cases_dict['total_cases'][i]}\n")

    def update_rewards(self):
        self.reward += (self.now - self.last_event_time) * (
                len(self.assigned_tasks) + len(self.unassigned_tasks))
        self.last_event_time = self.now

    def last_update_rewards(self):

        #don't account for cases that are not assigned yet
        self.reward += (self.now - self.last_event_time) * (
                len(self.assigned_tasks))
        self.last_event_time = self.now

    def reset(self):
        self.__init__(self.running_time, self.report, problem=self.problem, planner=None,
                      max_tasks=0)
        # Reset action tracking but keep the episode count and stats history

    def available_assignments(self):
        # Check if there are any available assignments
        for task in self.unassigned_tasks.values():
            # import pdb; pdb.set_trace()
            if len(set(self.available_resources).intersection(set(self.problem_resource_pool[task.task_type]))) > 0:
                return True
        return False

    def record_state_to_json(self, available_resources, unassigned_tasks, problem_resource_pool):
        state = {
            'available_resources': available_resources,
            'unassigned_tasks': [task.task_type for task in unassigned_tasks],
            'busy_resources': [resource[0] for resource in self.busy_resources],
            'mask': self.get_mask()[0].tolist(),
        }
        file_name = self.instance_file.split('/')[-1].split('.')[0]
        with open(f'states_{file_name}.json', 'a') as f:
            json.dump(state, f)
            #delimiter
            f.write('\n')
