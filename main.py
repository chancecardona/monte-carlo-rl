#!/bin/env python3
import numpy as np
np.bool = np.bool_

import json

import matplotlib.pyplot as plt

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
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.
import imageio

# Import our functions
from huggingface_utils import record_video
from reinforce import reinforce
from policy import Policy
from evaluate import evaluate_agent

if __name__ == '__main__':
    # Set the device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Set the environment
    env_id = "CartPole-v1"
    # Create the env
    env = gym.make(env_id)
    # Create the evaluation env
    eval_env = gym.make(env_id)
    
    # Get the state space and action space
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    
    print("_____OBSERVATION SPACE_____ \n")
    print("The State Space is: ", s_size)
    print("Sample observation", env.observation_space.sample()) # Get a random observation
    
    print("\n _____ACTION SPACE_____ \n")
    print("The Action Space is: ", a_size)
    print("Action Space Sample", env.action_space.sample()) # Take a random action

    # Set policy passing in the device
    debug_policy = Policy(s_size, a_size, 64, device).to(device)
    debug_policy.act(env.reset())

    # Load hyperparameters
    #with open('cartpole_hyperparameters.json', 'r') as f:
    #    # Load the JSON data as a dictionary
    #    cartpole_hyperparameters = json.load(f)
    cartpole_hyperparameters =  {
        "h_size": 16,
        "n_training_episodes": 1200,
        "n_evaluation_episodes": 15,
        "max_t": 1200,
        "gamma": 1.0,
        "lr": 0.8e-2,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size
    }


    # Create policy and place it to the device
    cartpole_policy = Policy(
        cartpole_hyperparameters["state_space"],
        cartpole_hyperparameters["action_space"],
        cartpole_hyperparameters["h_size"],
    ).to(device)
    cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

    scores = reinforce(
        env,
        cartpole_policy,
        cartpole_optimizer,
        cartpole_hyperparameters["n_training_episodes"],
        cartpole_hyperparameters["max_t"],
        cartpole_hyperparameters["gamma"],
        100,
    )

    evaluate_scores = evaluate_agent(
        eval_env, 
        cartpole_hyperparameters["max_t"], 
        cartpole_hyperparameters["n_evaluation_episodes"], 
        cartpole_policy
    )
    
    # Huggingface Hub
    upload = input("Upload to Huggingface hub? y/n:\n")
    if upload.lower() == "y":
        repo_id = "kismet163/ReinforceMonteCarlo" 
        push_to_hub(repo_id,
                    cartpole_policy, # The model we want to save
                    cartpole_hyperparameters, # Hyperparameters
                    eval_env, # Evaluation environment
                    video_fps=30
                    )
