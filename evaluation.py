import argparse
import json
import os
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf
import minari
from agents.ql_diffusion import AgentConfig
from agents.ql_diffusion import Diffusion_QL as Agent

torch.set_float32_matmul_precision("high")


def eval_policy(
    policy: Agent,
    eval_env: gym.Env,
    seed: int,
    eval_episodes: int = 10,
) -> Tuple[np.floating, np.floating, np.floating]:
    """
    Evaluates the policy over a number of episodes and returns metrics.
    Handles both Box and Dict observation spaces.
    """
    scores, episode_lengths = [], []
    for i in range(eval_episodes):
        state, _ = eval_env.reset(seed=seed + 100 + i)
        terminated, truncated = False, False
        ep_len, ep_ret = 0, 0.0

        while not (terminated or truncated):
            if isinstance(state, dict):
                obs_for_policy = state["observation"]
            else:
                obs_for_policy = state

            action = policy.sample_action(np.array(obs_for_policy))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            ep_ret += reward
            ep_len += 1

        scores.append(ep_ret)
        episode_lengths.append(ep_len)

    return np.mean(scores), np.std(scores), np.mean(episode_lengths)


def run_evaluation(run_dir: str):
    """
    Loads a trained agent and evaluates it on its corresponding environment.

    Args:
        run_dir (str): The path to the Hydra run directory, which should
                       contain '.hydra/config.yaml', 'actor.pth', and 'critic.pth'.
    """
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Hydra config not found at {config_path}. "
            "Please provide a valid Hydra run directory."
        )

    cfg = OmegaConf.load(config_path)
    print("--- Configuration Loaded ---")
    print(OmegaConf.to_yaml(cfg))

    print(f"Loading dataset '{cfg.env.name}' to recover environment...")
    dataset = minari.load_dataset(cfg.env.name)
    eval_env = dataset.recover_environment()

    if isinstance(eval_env.observation_space, gym.spaces.Dict):
        obs_space = eval_env.observation_space["observation"]
    else:
        obs_space = eval_env.observation_space

    state_dim = obs_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])
    print(f"Environment: {cfg.env.filename}, State Dim: {state_dim}, Action Dim: {action_dim}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent_cfg = AgentConfig(**cfg.agent)
    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        cfg=agent_cfg,
    )

    print(f"Loading model weights from: {run_dir}")
    agent.load_model(run_dir)
    agent.actor.eval()
    agent.critic.eval()
    print("--- Model Loaded ---")

    print("Running evaluation...")
    avg_reward, std_reward, avg_ep_len = eval_policy(
        agent, eval_env, cfg.seed, eval_episodes=cfg.eval_episodes
    )

    print("\n---------------------------------------")
    print(f"Evaluation over {cfg.eval_episodes} episodes:")
    print(f"Average Reward: {avg_reward:.3f} +/- {std_reward:.3f}")
    print(f"Average Episode Length: {avg_ep_len:.1f}")
    print("---------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Diffusion QL agent."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the Hydra run directory containing the model and params.json.",
    )
    args = parser.parse_args()
    run_evaluation(args.run_dir)
