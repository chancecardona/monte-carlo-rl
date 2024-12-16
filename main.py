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
from pixelcopter_policy import PixelcopterPolicy
from cartpole_policy import CartpolePolicy
from cartpole_agent import CartPoleMonteCarlo
from pixelcopter_agent import PixelCopterMonteCarlo
import optuna

# Objective function for Optuna
def cartpole_objective(trial: optuna.Trial):
    agent = CartPoleMonteCarlo(trial, device)
    agent.train()
    scores = agent.evaluate()
    return scores[0].mean()

def pixelcopter_objective(trial: optuna.Trial):
    agent = PixelCopterMonteCarlo(trial, device)
    agent.train()
    scores = agent.evaluate()
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
        study.optimize(cartpole_objective, n_trials=15, timeout=1600)
        trial = study.best_trial
        print("Finished Optuna optimization.")

        # Create agent with best hyperparameters and push to hub
        best_agent = CartPoleMonteCarlo(trial, device)
        best_agent.cartpole_hyperparameters.update(trial.params)
        print("Best agent params: ", best_agent.cartpole_hyperparameters)
        best_agent.train()
        best_agent.evaluate()
        
        # Huggingface Hub
        if args.upload:
            repo_id = "kismet163/ReinforceMonteCarlo" 
            push_to_hub(
                        repo_id,
                        best_agent.cartpole_policy, # The model we want to save
                        best_agent.cartpole_hyperparameters, # Hyperparameters
                        best_agent.eval_env, # Evaluation environment
                        video_fps=30
                        )


    elif args.environment == "pixelcopter":
        print("Running in Pixelcopter env.")
        
        # Optimize hyperparameters with OpTuna
        study = optuna.create_study(direction="maximize")
        study.optimize(pixelcopter_objective, n_trials=3, timeout=1200)
        trial = study.best_trial
        print("Finished Optuna optimization.")

        # Create agent with best hyperparameters and push to hub
        best_agent = PixelCopterMonteCarlo(trial, device)
        best_agent.pixelcopter_hyperparameters.update(trial.params)
        best_agent.train()
        best_agent.evaluate()

        repo_id = "kismet163/ReinforceMonteCarlo" 
        push_to_hub(
            repo_id,
            best_agent.pixelcopter_policy,  # The model we want to save
            best_agent.pixelcopter_hyperparameters,  # Hyperparameters
            best_agent.eval_env,  # Evaluation environment
            video_fps=30
        )
