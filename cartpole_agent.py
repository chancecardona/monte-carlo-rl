from cartpole_reinforce import reinforce
from cartpole_policy import CartpolePolicy
from evaluate import evaluate_agent
from optuna_hyperparameter_sampler import sample_cartpole_params
import torch.optim as optim

# Gym
import gym
import gym_pygame

class CartPoleMonteCarlo:
    def __init__(self, trial, device):
        self.env_id = "CartPole-v1"
        # Create the env
        self.env = gym.make(self.env_id, render_mode="rgb_array")
        # Create the evaluation env
        self.eval_env = gym.make(self.env_id, render_mode="rgb_array")
        
        # Get the state space and action space
        self.s_size = self.env.observation_space.shape[0]
        self.a_size = self.env.action_space.n
        
        print("_____OBSERVATION SPACE_____ \n")
        print("The State Space is: ", self.s_size)
        print("Sample observation", self.env.observation_space.sample()) # Get a random observation
        
        print("\n _____ACTION SPACE_____ \n")
        print("The Action Space is: ", self.a_size)
        print("Action Space Sample", self.env.action_space.sample()) # Take a random action

        # Set policy passing in the device
        self.debug_policy = CartpolePolicy(self.s_size, self.a_size, 64, device).to(device)
        self.debug_policy.act(self.env.reset())

        # Get Hyperparameters to make real policy
        # Sample with Optuna
        self.cartpole_hyperparameters = sample_cartpole_params(trial)
        self.cartpole_hyperparameters.update(
            {
                "env_id": self.env_id,
                "state_space": self.s_size,
                "action_space": self.a_size
            }
        )
        # Create actual policy passing it to the device
        self.cartpole_policy = CartpolePolicy(
            self.cartpole_hyperparameters["state_space"],
            self.cartpole_hyperparameters["action_space"],
            self.cartpole_hyperparameters["h_size"],
            device,
        ).to(device)

        self.cartpole_optimizer = optim.Adam(self.cartpole_policy.parameters(), lr=self.cartpole_hyperparameters["lr"])

    def train(self):
        scores = reinforce(
            self.env,
            self.cartpole_policy,
            self.cartpole_optimizer,
            self.cartpole_hyperparameters["n_training_episodes"],
            self.cartpole_hyperparameters["max_t"],
            self.cartpole_hyperparameters["gamma"],
            100,
        )
        return scores
    
    def evaluate(self):
        evaluate_scores = evaluate_agent(
            self.eval_env,
            self.cartpole_hyperparameters["max_t"],
            self.cartpole_hyperparameters["n_evaluation_episodes"],
            self.cartpole_policy
        )
        return evaluate_scores 
