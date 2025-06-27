# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import os
import torch
import csv
import json
import minari
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from data_sampler import Data_Sampler
from hydra.core.hydra_config import HydraConfig
from agents.ql_diffusion import Diffusion_QL as Agent
from agents.ql_diffusion import AgentConfig
from typing import Tuple
import wandb

torch.set_float32_matmul_precision("high")


def eval_policy(policy: Agent, eval_env: gym.Env, seed: int, eval_episodes: int = 10) -> Tuple[np.floating, np.floating, np.floating]:
    """
    Evaluates the policy over a number of episodes and returns metrics.

    Args:
        policy (Agent): The agent/policy to evaluate.
        eval_env (gym.Env): The Gymnasium environment to evaluate on.
        seed (int): The random seed for environment resets.
        eval_episodes (int, optional): The number of episodes to run. Defaults to 10.

    Returns:
        A tuple containing:
        - The average reward over all episodes.
        - The standard deviation of the reward.
        - The average length of the episodes.
    """
    ep_length = [0] * eval_episodes
    scores = []
    for i in range(eval_episodes):
        state, info = eval_env.reset(seed=seed + 100 + i)
        terminated, truncated = False, False
        traj_return = 0.0

        while not (terminated or truncated):
            if isinstance(state, dict):
                state = state['observation']
            ep_length[i] += 1
            action = policy.sample_action(np.array(state))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    return avg_reward, std_reward, np.mean(ep_length)


def train_agent(cfg: DictConfig) -> None:
    """
    The main training loop for the Diffusion QL agent.

    Args:
        cfg (DictConfig): The Hydra configuration object.
    """
    
    results_dir = HydraConfig.get().runtime.output_dir
    device = f"cuda:{cfg.device_id}" if torch.cuda.is_available() else "cpu"
    minari_env_name = cfg.env.name
 
    dataset = minari.load_dataset(minari_env_name)    
    env = dataset.recover_environment()
    if isinstance(env.observation_space, gym.spaces.Dict): obs_space = env.observation_space['observation']
    else: obs_space = env.observation_space
    state_dim = obs_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    data_sampler = Data_Sampler(dataset, device, cfg.reward_tune)
    
    agent_cfg = AgentConfig(**cfg.agent)
    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        cfg=agent_cfg
    )

    agent.actor.model = torch.compile(agent.actor.model)
    agent.critic = torch.compile(agent.critic)

    training_iters = 0
    max_timesteps = cfg.num_epochs * cfg.num_steps_per_epoch
    pbar = tqdm(total=max_timesteps, desc="Total Training Progress")

    while training_iters < max_timesteps:
        iterations = int(cfg.eval_freq * cfg.num_steps_per_epoch)
        loss_metric = agent.train(
            data_sampler,
            iterations=iterations,
            batch_size=cfg.batch_size,
            progress_bar=pbar
        )

        # W&B Logging
        for i in range(iterations):
            step = training_iters + i
            log_dict = {
                "train/actor_loss": loss_metric["actor_loss"][i],
                "train/critic_loss": loss_metric["critic_loss"][i],
                "train/bc_loss": loss_metric["bc_loss"][i],
                "train/ql_loss": loss_metric["ql_loss"][i],
                "train/avg_q_batch": loss_metric["avg_q_batch"][i],
                "train/avg_q_policy": loss_metric["avg_q_policy"][i],
                "train/actor_grad_norm": loss_metric["actor_gn"][i],
                "train/critic_grad_norm": loss_metric["critic_gn"][i],
                "step": step,
                "epoch": int(step // cfg.num_steps_per_epoch),
            }
            wandb.log(log_dict)

        training_iters += iterations

        # Evaluation
        eval_env = dataset.recover_environment()
        eval_reward, eval_std, eval_len = eval_policy(
            agent, eval_env, cfg.seed, eval_episodes=cfg.eval_episodes
        )

        wandb.log(
            {
                "eval/avg_reward": eval_reward,
                "eval/std_reward": eval_std,
                "eval/avg_ep_length": eval_len,
                "step": training_iters,
                "epoch": int(training_iters // cfg.num_steps_per_epoch),
            }
        )

        print(
            f"Epoch: {int(training_iters // int(cfg.num_steps_per_epoch))}, Step: {training_iters}, Avg Reward: {eval_reward:.3f}\n"
        )

        if cfg.save_best_model:            
            agent.save_model(
                results_dir, None
            )  

    pbar.close()
    wandb.finish()
    print("Training Finished")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    results_dir = HydraConfig.get().runtime.output_dir
    wandb.init(
        project="Diffusion_QL",
        group=f"{cfg.env.name}",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        dir=results_dir
    )

    print(OmegaConf.to_yaml(cfg))

    print(f"Saving Location: {results_dir}\n")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_agent(cfg)


if __name__ == "__main__":
    main()
