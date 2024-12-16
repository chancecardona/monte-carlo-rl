from typing import Any, Dict
import optuna

def sample_cartpole_params(trial: optuna.Trial) -> Dict[str, Any]:
    h_size = 16 # Keep constant for now
    n_training_episodes = 1500 - trial.suggest_int("n_training_episodes", 100, 1000, log=True) # Subtract with Log since we want more samples near more training eps
    n_evaluation_episodes = 55 - trial.suggest_int("n_evaluation_episodes", 5, 45, log=True)
    max_t = trial.suggest_categorical("max_t", [200, 500, 900, 1000, 1100, 1200, 1500])
    gamma = 1 - trial.suggest_float("gamma", 0.0001, 0.9, log=True)
    lr = trial.suggest_float("lr", 0.1e-3, 0.2, log=True)

    cartpole_hyperparameters = {
        "h_size": h_size,
        "n_training_episodes": n_training_episodes,
        "n_evaluation_episodes": n_evaluation_episodes,
        "max_t": max_t,
        "gamma": gamma,
        "lr": lr,
    }
    return cartpole_hyperparameters

def sample_pixelcopter_params(trial: optuna.Trial) -> Dict[str, Any]:
    h_size = 64 # Keep constant for now
    n_training_episodes = 40000 - trial.suggest_int("n_training_episodes", 1000, 30000, log=True) # Subtract with Log since we want more samples near more training eps
    n_evaluation_episodes = 20 - trial.suggest_int("n_evaluation_episodes", 5, 14, log=True)
    max_t = trial.suggest_categorical("max_t", [5000, 10000, 15000, 20000])
    gamma = 1 - trial.suggest_float("gamma", 0.0001, 0.9, log=True)
    lr = trial.suggest_float("lr", 0.1e-4, 0.1, log=True)

    pixelcopter_hyperparameters = {
        "h_size": h_size,
        "n_training_episodes": n_training_episodes,
        "n_evaluation_episodes": n_evaluation_episodes,
        "max_t": max_t,
        "gamma": gamma,
        "lr": lr,
    }
    return pixelcopter_hyperparameters
