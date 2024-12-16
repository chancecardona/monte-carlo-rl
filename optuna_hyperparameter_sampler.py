from typing import Any, Dict
import optuna

def sample_cartpole_params(trial: optuna.Trial) -> Dict[str, Any]:
    h_size = 16 # Keep constant for now
    n_training_episodes = trial.suggest_int("n_training_episodes", 400, 1500, log=False) # Subtract with Log since we want more samples near more training eps
    n_evaluation_episodes = trial.suggest_int("n_evaluation_episodes", 55, 400, log=True)
    print("eval episodes", n_evaluation_episodes)
    max_t = trial.suggest_categorical("max_t", [896, 960, 1024, 1536, 2048])
    gamma = trial.suggest_float("gamma", 1e-3, 0.999, log=True)
    lr = trial.suggest_float("lr", 1e-4, 0.2, log=True)

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
    n_training_episodes = 30000 - trial.suggest_int("n_training_episodes", 1000, 20000, log=True) # Subtract with Log since we want more samples near more training eps
    n_evaluation_episodes = 1000 - trial.suggest_int("n_evaluation_episodes", 5, 80, log=True)
    max_t = trial.suggest_categorical("max_t", [5000, 10000, 15000, 20000])
    gamma = trial.suggest_float("gamma", 0.0001, 0.999, log=True)
    lr = trial.suggest_float("lr", 1e-4, 0.1, log=True)

    pixelcopter_hyperparameters = {
        "h_size": h_size,
        "n_training_episodes": n_training_episodes,
        "n_evaluation_episodes": n_evaluation_episodes,
        "max_t": max_t,
        "gamma": gamma,
        "lr": lr,
    }
    return pixelcopter_hyperparameters
