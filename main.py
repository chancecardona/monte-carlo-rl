import numpy as np

from collections import deque

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

    debug_policy = Policy(s_size, a_size, 64).to(device)
    debug_policy.act(env.reset())

    # Create policy and place it to the device
    cartpole_policy = Policy(
        cartpole_hyperparameters["state_space"],
        cartpole_hyperparameters["action_space"],
        cartpole_hyperparameters["h_size"],
    ).to(device)
    cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

    scores = reinforce(
        cartpole_policy,
        cartpole_optimizer,
        cartpole_hyperparameters["n_training_episodes"],
        cartpole_hyperparameters["max_t"],
        cartpole_hyperparameters["gamma"],
        100,
    )

    evaluate_agent(
        eval_env, 
        cartpole_hyperparameters["max_t"], 
        cartpole_hyperparameters["n_evaluation_episodes"], 
        cartpole_policy
    )
