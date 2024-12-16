from typing import Any, Dict
import optuna

def sample_cartpole_params(trial: optuna.Trial) -> Dict[str, Any]:
    h_size = 16 # Keep constant for now
    n_training_episodes = trial.suggest_int("n_training_episodes", 10, 500, log=False)
    n_evaluation_episodes = trial.suggest_int("n_evaluation_episodes", 5, 50, log=False)
    max_t = trial.suggest_categorical("max_t", [200, 500, 1000, 1100, 1200, 1500])
    gamma = trial.suggest_float("gamma", 0.5, 1.5, log=True)
    lr = trial.suggest_float("lr", 0.1e-3, 0.5, log=True)

    cartpole_hyperparameters = {
        "h_size": h_size,
        "n_training_episodes": n_training_episodes,
        "n_evaluation_episodes": n_evaluation_episodes,
        "max_t": max_t,
        "gamma": gamma,
        "lr": lr,
    }
    return cartpole_hyperparameters
