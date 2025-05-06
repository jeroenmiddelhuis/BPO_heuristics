import optuna
from train_policy import make_env
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO
import numpy as np

def objective(trial):
    # Suggest hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_neurons = trial.suggest_categorical("n_neurons", [64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    learning_rate = trial.suggest_categorical("learning_rate", [3e-5, 3e-4, 3e-3])
    gamma = trial.suggest_categorical("gamma", [0.99, 0.999])
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.98)
    ent_coef = trial.suggest_float("ent_coef", 0, 0.1)
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20])

    # Build net_arch
    net_arch = dict(
        pi=[n_neurons for _ in range(n_layers)],
        vf=[n_neurons for _ in range(n_layers)]
    )

    # Create environment
    config_type = "slow_server"
    nr_cases = 500
    env = make_env(config_type, nr_cases)

    # Create model with suggested hyperparameters
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        n_epochs=n_epochs,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=0,
    )

    # Train for a small number of timesteps for tuning
    model.learn(total_timesteps=20000)

    # Re-create environment for evaluation to avoid training artifacts
    eval_env = make_env(config_type, nr_cases)
    eval_simulator = eval_env.simulator

    nr_eval_episodes = 10
    cycle_times = []

    for _ in range(nr_eval_episodes):
        obs, _ = eval_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, action_masks=eval_env.action_masks(), deterministic=True)
            obs, reward, done, _, _ = eval_env.step(np.int32(action))

        # Compute average cycle time for this episode
        if len(eval_simulator.completed_cases) == nr_cases:
            avg_cycle_time = sum(case.cycle_time for case in eval_simulator.completed_cases) / len(eval_simulator.completed_cases)
            cycle_times.append(avg_cycle_time)
        else:
            # Penalize if not all cases completed
            cycle_times.append(float("inf"))

    # Return mean cycle time over evaluation episodes
    mean_cycle_time = np.mean(cycle_times) if cycle_times else float("inf")
    return mean_cycle_time

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    print("Best hyperparameters:", study.best_params)