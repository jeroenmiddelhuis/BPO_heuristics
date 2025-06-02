from rw_simulator import Simulator
from rw_gym_env import Environment
import os

# possible values: ppo, spt, fifo, bayes, random
problem_name = "bpi2017" #possible values bpi2012, bpi2017, bpi2018, consulta, production
problem_type = 'regenerated'  # possible values original, regenerated

multiple_models = ['spt'] #['random', 'fifo_process', 'spt', 'ppo']
num_replicates = 1

if problem_name == 'toloka':
    instance_file = "./data/toloka_problem.pkl"
    n_edges = 132
elif problem_name == 'fines':
    instance_file = "./data/fines_problem.pkl"
    n_edges = 53
elif problem_name == 'bpi2017':
    n_edges = 573
    if problem_type == 'original':
        instance_file = "./BPI Challenge 2017 - instance.pickle"
    else:
        instance_file = "./data/bpi2017_problem.pkl"
elif problem_name == 'bpi2012':
    instance_file = "./data/bpi2012_problem.pkl"
    n_edges = 199
elif problem_name == 'bpi2018':
    instance_file = "./data/bpi2018_problem.pkl"
    n_edges = 479
elif problem_name == 'consulta':
    instance_file = "./data/consulta.pkl"
    n_edges = 435
elif problem_name == 'production':
    instance_file = "data/production.pkl"
    n_edges = 76
elif problem_name == 'microsoft':
    instance_file = "data/microsoft.pkl"
    n_edges = 55
else:
    raise Exception("Invalid problem name")

my_planner = None

simulator = Simulator(nr_cases=1000,
                        planner=my_planner, instance_file=instance_file)

def evaluate(config_type, policy, nr_cases, nr_episodes):
    # Initialize simulator and environment
    simulator = Simulator(config_type, nr_cases)
    env = Environment(simulator)
    
    cycle_times = []

    for _ in range(nr_episodes):
        env.reset()
        done = False
        step_count = 0

        while not done:
            action = policy(simulator)
            obs, reward, done, _, _ = env.step(action)
            step_count += 1

            # if step_count % 100 == 0:
            #     queue_lengths = simulator.get_queue_lengths_per_task_type()
            #     avg_queue_length = np.mean(list(queue_lengths.values()))
            #     print(f"Step {step_count}: Average queue length = {avg_queue_length:.2f}, Per task type: {queue_lengths}")

        assert len(simulator.completed_cases) == nr_cases, f"Expected {nr_cases} completed cases, but got {len(simulator.completed_cases)}"
        cycle_times.append(sum(case.cycle_time for case in simulator.completed_cases) / len(simulator.completed_cases))
        print(sum(case.cycle_time for case in simulator.completed_cases) / len(simulator.completed_cases))

    # Write results to file
    # results_dir = f'results/{config_type}'
    # os.makedirs(results_dir, exist_ok=True)
    # results_file = f'{results_dir}/{config_type}_{policy.__name__}.txt'
    # with open(results_file, 'w') as f:
    #     f.write("cycle_time\n")
    #     for cycle_time in cycle_times:
    #         f.write(f"{cycle_time}\n")

