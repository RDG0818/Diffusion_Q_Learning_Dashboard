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
from utils.data_sampler import Data_Sampler
from hydra.core.hydra_config import HydraConfig
from agents.ql_diffusion import Diffusion_QL as Agent
from agents.ql_diffusion import AgentConfig

torch.set_float32_matmul_precision("high")


def train_agent(results_dir: str, cfg: DictConfig) -> None:
    minari_env_name = f"mujoco/{cfg.env.name}/{cfg.quality.name}-v0"
    gym_env_name = f"{cfg.env.name.capitalize()}-v5"
    if "cheetah" in gym_env_name:
        gym_env_name = "HalfCheetah-v5"

    device = f"cuda:{cfg.device_id}" if torch.cuda.is_available() else "cpu"

    env = gym.make(gym_env_name)
    state_dim, action_dim, max_action = (
        env.observation_space.shape[0],
        env.action_space.shape[0],
        float(env.action_space.high[0]),
    )

    dataset = minari.load_dataset(minari_env_name)
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

    # Logger
    progress_csv_path = os.path.join(results_dir, "progress.csv")
    csv_header = [
        "step",
        "avg_reward",
        "std_reward",
        "avg_ep_length",
        "actor_loss",
        "critic_loss",
        "bc_loss",
        "ql_loss",
        "avg_q_batch",
        "avg_q_policy",
        "actor_grad_norm",
        "critic_grad_norm",
    ]
    if not os.path.exists(progress_csv_path):
        with open(progress_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    training_iters = 0
    max_timesteps = cfg.num_epochs * cfg.num_steps_per_epoch
    with tqdm(total=max_timesteps, desc="Total Training Progress") as pbar:
        while training_iters < max_timesteps:
            iterations = int(cfg.eval_freq * cfg.num_steps_per_epoch)
            loss_metric = agent.train(
                data_sampler,
                iterations=iterations,
                batch_size=cfg.batch_size,
                progress_bar=pbar,
            )
            training_iters += iterations
            curr_epoch = int(training_iters // int(cfg.num_steps_per_epoch))

            # Evaluation
            eval_env = gym.make(gym_env_name)
            eval_res, eval_res_std, avg_ep_length = eval_policy(
                agent, eval_env, cfg.seed, eval_episodes=cfg.eval_episodes
            )

            # Logging
            log_row = [
                training_iters,
                eval_res,
                eval_res_std,
                avg_ep_length,
                np.mean(loss_metric["bc_loss"]),
                np.mean(loss_metric["ql_loss"]),
                np.mean(loss_metric["actor_loss"]),
                np.mean(loss_metric["critic_loss"]),
                np.mean(loss_metric["avg_q_batch"]),
                np.mean(loss_metric["avg_q_policy"]),
                np.mean(loss_metric["actor_gn"]),
                np.mean(loss_metric["critic_gn"]),
            ]

            with open(progress_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(log_row)

            print(
                f"Epoch: {curr_epoch}, Step: {training_iters}, Avg Reward: {eval_res:.3f}, Actor Loss: {np.mean(loss_metric['actor_loss']):.3f}, Critic Loss: {np.mean(loss_metric['critic_loss']):.3f}\n"
            )

            if cfg.save_best_model:
                agent.save_model(
                    results_dir, None
                )  # Can change to curr_epoch if you want all


# Runs policy for X episodes and returns average reward
def eval_policy(policy, eval_env, seed, eval_episodes=10):
    ep_length = [0] * eval_episodes
    scores = []
    for i in range(eval_episodes):
        state, info = eval_env.reset(seed=seed + 100 + i)
        terminated, truncated = False, False
        traj_return = 0.0

        while not (terminated or truncated):
            ep_length[i] += 1
            action = policy.sample_action(np.array(state))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    return avg_reward, std_reward, np.mean(ep_length)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    results_dir = HydraConfig.get().runtime.output_dir
    print(f"Saving Location: {results_dir}\n")

    minari_env_name = f"mujoco/{cfg.env.name}/{cfg.quality.name}-v0"
    gym_env_name = f"{cfg.env.name.capitalize()}-v5"
    if "cheetah" in gym_env_name:
        gym_env_name = "HalfCheetah-v5"

    flat_params = OmegaConf.to_container(cfg, resolve=True)
    flat_params["minari_env_name"] = minari_env_name
    flat_params["gym_env_name"] = gym_env_name

    params_path = os.path.join(results_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(flat_params, f, indent=4)
    print(f"Parameters saved to {params_path}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_agent(results_dir, cfg)


if __name__ == "__main__":
    main()
