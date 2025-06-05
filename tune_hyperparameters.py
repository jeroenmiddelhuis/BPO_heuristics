import optuna
from rw_train_policy import make_env
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO
import numpy as np
import json
import os

def objective(trial):
    # Suggest hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_neurons = trial.suggest_categorical("n_neurons", [64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [4096, 8192, 16384])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    learning_rate = trial.suggest_categorical("learning_rate", [3e-5, 3e-4, 3e-3])
    gamma = trial.suggest_categorical("gamma", [0.99, 0.999, 1.0])
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.98)
    ent_coef = trial.suggest_float("ent_coef", 0, 0.1)
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20])

    # Build net_arch
    net_arch = dict(
        pi=[n_neurons for _ in range(n_layers)],
        vf=[n_neurons for _ in range(n_layers)]
    )

    # Create environment
    problem_instance = "bpi2017"
    nr_cases = 1000
    env = make_env(problem_instance, nr_cases)

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
        verbose=1,
    )

    # Train for a small number of timesteps for tuning
    model.learn(total_timesteps=1000000)
    # Create directory for saving models if it doesn't exist
    os.makedirs("hyperparameter_tuning/models", exist_ok=True)
    model.save(f"hyperparameter_tuning/models/{trial.number}_model")

    # Re-create environment for evaluation to avoid training artifacts
    eval_env = make_env(problem_instance, nr_cases)

    nr_eval_episodes = 100
    cycle_times = []

    for _ in range(nr_eval_episodes):
        obs, _ = eval_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, action_masks=eval_env.action_masks(), deterministic=True)
            obs, reward, done, _, _ = eval_env.step(np.int32(action))

        # Compute average cycle time for this episode
        avg_cycle_time = eval_env.simulator.total_cycle_time / eval_env.simulator.n_finalized_cases
        cycle_times.append(avg_cycle_time)


    # Return mean cycle time over evaluation episodes
    mean_cycle_time = np.mean(cycle_times) if cycle_times else float("inf")
    print(f"Trial {trial.number} - Mean Cycle Time: {mean_cycle_time:.2f}")
    print(f"Hyperparameters: {trial.params}")
    print(f"Trial Value: {mean_cycle_time:.2f}")
    return mean_cycle_time

if __name__ == "__main__":
    os.makedirs("hyperparameter_tuning/", exist_ok=True)
    # Save study to a SQLite database file
    study = optuna.create_study(
        direction="minimize",
        study_name="ppo_hyperparam_tuning",
        storage="sqlite:///hyperparameter_tuning/ppo_hyperparam_tuning.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100, n_jobs=16)
    print("Best hyperparameters:", study.best_params)

    # Save all trial results to a JSON file
    all_trials = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state)
        }
        for t in study.trials
    ]

    with open("hyperparameter_tuning/all_trials.json", "w") as f:
        json.dump(all_trials, f, indent=4)

    # Save best hyperparameters to a JSON file
    with open("hyperparameter_tuning/best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f, indent=4)