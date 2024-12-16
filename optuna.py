from typing import Any, Dict

def sample_cartpole_params(trial: optuna.Trial) -> Dict[str, Any]:
    h_size = 16 # Keep constant for now
    n_training_episodes = trial.suggest_int("n_training_episodes", 10, 500, log=True)
    n_evaluation_episodes = trial.suggest_int("n_evaluation_episodes", 5, 50, log=True)
    max_t = trial.suggest_int("max_t", 200, 2500, log=True)
    gamma = trial.suggest_float("gamma", 0.5, 1.5, log=True)
    lr = trial.suggest_float("lr", 0.1e-3, 0.5, log=True)

    cartpole_hyperparameters = {
        "h_size": h_size,
        "n_training_episodes": n_training_episodes,
        "n_evaluation_episodes": n_evaluation_episodes,
        "max_t": max_t,
        "gamma": gamma,
        "lr": lr,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size
    }
    return cartpole_hyperparameters
