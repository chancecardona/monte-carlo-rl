#!/bin/env python3
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
import gym
import gym_pygame

# Hugging Face Hub
from huggingface_hub import notebook_login
import imageio

# Import our functions
from huggingface_utils import record_video, push_to_hub
from reinforce import reinforce
from pixelcopter_policy import PixelcopterPolicy
from cartpole_policy import CartpolePolicy
from evaluate import evaluate_agent

from cartpole_agent import CartPoleMonteCarlo
import optuna

# Objective function for Optuna
def cartpole_objective(trial: optuna.Trial):
    agent = CartPoleMonteCarlo(trial, device)
    agent.train()
    scores = agent.evaluate()
    #import pdb; pdb.set_trace()
    return scores[0].mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str, choices=["cartpole", "pixelcopter"], default="cartpole",
                        help="Environment to train and test RL model in, will auto-select policy.")
    parser.add_argument("-u", "--upload", type=bool, default=True,
                        help="Upload to Huggingface hub?")
    args = parser.parse_args()

    # Set the device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Utilising device:", device)
    print("Available devices:", torch.backends)
    
    # Set the environment
    if args.environment == "cartpole":
        print("Running in Cartpole env.")
        # Optimize hyperparameters with OpTuna
        study = optuna.create_study(direction="maximize")
        study.optimize(cartpole_objective, n_trials=100)
        trial = study.best_trial
        print("Finished Optuna optimization.")
        print("Best Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # Create agent with best hyperparameters and push to hub
        best_agent = CartPoleMonteCarlo(trial, device)
        best_agent.cartpole_hyperparameters = trial.params
        best_agent.train()
        best_agent.evaluate()
        
        # Huggingface Hub
        if args.upload:
            repo_id = "kismet163/ReinforceMonteCarlo" 
            push_to_hub(env,
                        repo_id,
                        best_agent.cartpole_policy, # The model we want to save
                        best_agent.cartpole_hyperparameters, # Hyperparameters
                        best_agent.eval_env, # Evaluation environment
                        video_fps=30
                        )


    elif args.environment == "pixelcopter":
        print("Running in Pixelcopter env.")
        env_id = "Pixelcopter-PLE-v0"
        env = gym.make(env_id)
        eval_env = gym.make(env_id)
        s_size = env.observation_space.shape[0]
        a_size = env.action_space.n
        
        print("_____OBSERVATION SPACE_____ \n")
        print("The State Space is: ", s_size)
        print("Sample observation", env.observation_space.sample())  # Get a random observation
        
        print("\n _____ACTION SPACE_____ \n")
        print("The Action Space is: ", a_size)
        print("Action Space Sample", env.action_space.sample())  # Take a random action

        pixelcopter_hyperparameters = {
            "h_size": 64,
            "n_training_episodes": 50000,
            "n_evaluation_episodes": 10,
            "max_t": 10000,
            "gamma": 0.99,
            "lr": 1e-4,
            "env_id": env_id,
            "state_space": s_size,
            "action_space": a_size,
        }
        # Create policy and place it to the device
        # torch.manual_seed(50)
        pixelcopter_policy = PixelcopterPolicy(
            pixelcopter_hyperparameters["state_space"],
            pixelcopter_hyperparameters["action_space"],
            pixelcopter_hyperparameters["h_size"],
            device,
        ).to(device)
        pixelcopter_optimizer = optim.Adam(pixelcopter_policy.parameters(), lr=pixelcopter_hyperparameters["lr"])
        
        scores = reinforce(
            env,
            pixelcopter_policy,
            pixelcopter_optimizer,
            pixelcopter_hyperparameters["n_training_episodes"],
            pixelcopter_hyperparameters["max_t"],
            pixelcopter_hyperparameters["gamma"],
            1000,
        )

        repo_id = "kismet163/ReinforceMonteCarlo" 
        push_to_hub(
            env,
            repo_id,
            pixelcopter_policy,  # The model we want to save
            pixelcopter_hyperparameters,  # Hyperparameters
            eval_env,  # Evaluation environment
            video_fps=30
        )
