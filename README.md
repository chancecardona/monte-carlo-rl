## Monte Carlo RL using Policy Gradient Diffusion

### Installation using UV
```bash
uv venv
uv pip install -r requirements.txt --break-system-packages
```

### To push to huggingface
`huggingface-cli login` after creating an identity token at huggingface.co


## Running
Cartpole Environment (default)
```bash
python3 main.py
```

Pixelcopter Environment
```bash
python3 main.py -e pixelcopter
```

`-u False` to prevent uploading to huggingface (which is the default)

## File Structure
This repo uses Optuna to optimize the trial hyperparameters for the agent.
- The NN architecture (layers, activation funcs, etc) for the agent (dependent on the environment) is in `<env_name>_policy.py`.  
- The NN model class definition is in `<env_name>_agent.py`  which contains the Train, Evaluate loops.  
- `optuna_hyperparameter_sampler.py` contains the hyperparameter definitions for Optuna.
- `reinforce.py` is the Monte Carlo RL algorithm
- `evaluate.py` accumalates the reward of the agent over the episodes and outputs the mean and std of the reward.
- `huggingface_utils.py` is the utilities to push to huggingface and record videos.
