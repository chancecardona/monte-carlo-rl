import numpy as np
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from pathlib import Path
import datetime
import json
import imageio

import tempfile
import time

import os

# Our code
from evaluate import evaluate_agent

def record_video(env, policy, out_directory, fps=30):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    done = False
    state = env.reset()
    img = env.render()
    images.append(img)
    while not done:
        # Take the action (index) that have the maximum expected future reward given that state
        action, _ = policy.act(state)
        state, reward, done, trunc, info = env.step(action) # We directly put next_state = state for recording logic
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)

def push_to_hub(
                repo_id,
                model,
                hyperparameters,
                eval_env,
                video_fps=30
                ):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It pushes everything to the Hub
    
    :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
    :param model: the pytorch model we want to save
    :param hyperparameters: training hyperparameters
    :param eval_env: evaluation environment
    :param video_fps: how many frame per seconds to record our video replay
    """
    
    _, repo_name = repo_id.split("/")
    api = HfApi()
    
    # Step 1: Create the repo
    repo_url = api.create_repo(
          repo_id=repo_id,
          exist_ok=True,
    )
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_directory = Path(tmpdirname)
        
        # Step 2: Save the model
        torch.save(model, local_directory / "model.pt")
        
        # Step 3: Save the hyperparameters to JSON
        with open(local_directory / "hyperparameters.json", "w") as outfile:
          json.dump(hyperparameters, outfile)
        
        # Step 4: Evaluate the model and build JSON
        mean_reward, std_reward = evaluate_agent(eval_env,
                                                hyperparameters["max_t"],
                                                hyperparameters["n_evaluation_episodes"],
                                                model)
        # Get datetime
        eval_datetime = datetime.datetime.now()
        eval_form_datetime = eval_datetime.isoformat()
        
        #env_name = eval_env.env_id
        env_name = hyperparameters["env_id"]
        evaluate_data = {
              "env_id": env_name,
              "mean_reward": mean_reward,
              "n_evaluation_episodes": hyperparameters["n_evaluation_episodes"],
              "eval_datetime": eval_form_datetime,
        }
        
        # Write a JSON file
        with open(local_directory / "results.json", "w") as outfile:
            json.dump(evaluate_data, outfile)
        
        # Step 5: Create the model card 
        metadata = {}
        metadata["tags"] = [
              env_name,
              "reinforce",
              "reinforcement-learning",
              "custom-implementation",
              "deep-rl-class"
          ]
        
        # Add metrics
        eval = metadata_eval_result(
            model_pretty_name=repo_name,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=env_name,
            dataset_id=env_name,
          )
        
        # Merges both dictionaries
        metadata = {**metadata, **eval}
        
        model_card = f"""
        # **RL Monte Carlo** Agent playing **{env_name}**
        This is a trained model of a **RL Monte Carlo** agent playing **{env_name}** .
        To learn to use this model and train yours check Unit 4 of the Deep Reinforcement Learning Course: https://huggingface.co/deep-rl-course/unit4/introduction
        """
    
        readme_path = local_directory / "README.md"
        readme = ""
        if readme_path.exists():
            with readme_path.open("r", encoding="utf8") as f:
                readme = f.read()
        else:
            readme = model_card
    
        with readme_path.open("w", encoding="utf-8") as f:
            f.write(readme)
    
        # Save our metrics to Readme metadata
        metadata_save(readme_path, metadata)
    
        # Step 6: Record a video
        video_path =  local_directory / "replay.mp4"
        record_video(eval_env, model, video_path, video_fps)
    
        # Step 7. Push everything to the Hub
        attempt = 0
        max_delay = 100
        while attempt < max_delay:   
            try:
                api.upload_folder(
                      repo_id=repo_id,
                      folder_path=local_directory,
                      path_in_repo=".",
                )
                print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")
                return
            except requests.exceptions.ConnectionError as e:
                attempt += 1
                delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)  # Exponential backoff
                print(f"Attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            except requests.exceptions.RequestException as e:
                # Handle other types of request exceptions
                print(f"Request failed: {e}")
                break  # Break if it's a different error, no need to retry
        print("Max retries failed, exiting failure.")
        return
