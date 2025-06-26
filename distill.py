import torch
import torch.nn as nn
import numpy as np
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import minari
import json
from agents.ql_diffusion import Diffusion_QL
from utils.data_sampler import Data_Sampler
import time
import csv
import gymnasium as gym
import pandas as pd
from tqdm import tqdm


# Evaluate policy with progress bars
def eval_policy(policy, eval_env, seed, eval_episodes=10):
    rewards, lengths, times = [], [], []
    for i in tqdm(range(eval_episodes), desc="Evaluating Episodes", unit="ep"):
        state, _ = eval_env.reset(seed=seed + 100 + i)
        traj_return, step = 0.0, 0
        start = time.time()
        while True:
            action = policy.sample_action(np.array(state))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            traj_return += reward
            step += 1
            if terminated or truncated:
                break
        end = time.time()
        rewards.append(traj_return)
        lengths.append(step)
        times.append(end - start)
    return rewards, lengths, times


# Student network with three hidden layers
class StudentPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),  # third hidden
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state):
        return self.net(state) * self.max_action

    def sample_action(self, state: np.ndarray):
        state_t = (
            torch.from_numpy(state)
            .float()
            .unsqueeze(0)
            .to(next(self.parameters()).device)
        )
        with torch.no_grad():
            return self.forward(state_t).cpu().numpy()[0]


# Load teacher agent
def load_agent_params(model_dir: str, params_path: str):
    with open(params_path, "r") as f:
        params = json.load(f)

    state_dim = params["env"]["observation_space"]
    action_dim = params["env"]["action_space"]
    gym_env_name = params["gym_env_name"]
    max_action = params.get("max_action", 1.0)
    device = (
        f"cuda:{params.get('device_id', 0)}" if torch.cuda.is_available() else "cpu"
    )

    agent = Diffusion_QL(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=params.get("discount", 0.99),
        tau=params.get("tau", 0.005),
        max_q_backup=params.get("max_q_backup", False),
        beta_schedule=params.get("beta_schedule", "linear"),
        n_timesteps=params.get("T", 5),
        eta=params.get("eta", 1.0),
        lr=params.get("lr", 3e-4),
        lr_decay=params.get("lr_decay", False),
        lr_maxt=params.get("num_epochs", 100),
        grad_norm=params.get("gn", 5.0),
    )
    agent.load_model(model_dir)
    agent.actor.eval()
    return agent, device, params


@hydra.main(config_path="configs", config_name="distill", version_base="1.3")
def distill(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # Load teacher
    teacher_p = os.path.join(cfg.teacher_model_dir, "params.json")
    teacher, device, teacher_params = load_agent_params(
        cfg.teacher_model_dir, teacher_p
    )

    # Prepare data
    minari_env = f"mujoco/{cfg.env.name}/{cfg.quality.name}-v0"
    print(f"Loading dataset {minari_env}...")
    dataset = minari.load_dataset(minari_env)
    sampler = Data_Sampler(dataset, device, cfg.reward_tune)

    # Build student
    student = StudentPolicy(
        state_dim=teacher_params["env"]["observation_space"],
        action_dim=teacher_params["env"]["action_space"],
        max_action=teacher_params.get("max_action", 1.0),
    ).to(device)
    student = torch.compile(student)

    optimizer = torch.optim.Adam(
        student.parameters(), lr=cfg.distill_lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.distill_epochs
    )
    loss_fn = nn.SmoothL1Loss()

    # DAgger buffer
    dagger_states = []
    dagger_actions = []

    # Initial distillation
    for ep in tqdm(range(1, cfg.distill_epochs + 1), desc="Distillation", unit="ep"):
        states, *_ = sampler.sample(cfg.distill_batch_size)
        with torch.no_grad():
            target_acts = teacher.actor.sample(states)
        pred = student(states)
        loss = loss_fn(pred, target_acts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # DAgger iterations
    env = gym.make(teacher_params["gym_env_name"])
    for itr in range(1, cfg.dagger_iters + 1):
        print(f"DAgger Iteration {itr}/{cfg.dagger_iters}")
        # Rollout student to collect states
        for _ in range(cfg.dagger_rollouts):
            s, _ = env.reset()
            done = False
            while not done:
                a = student.sample_action(np.array(s))
                dagger_states.append(s.copy())
                # query teacher
                with torch.no_grad():
                    t_a = teacher.actor.sample(
                        torch.from_numpy(s).float().unsqueeze(0).to(device)
                    )
                dagger_actions.append(t_a.cpu().numpy()[0])
                s, _, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
        # Convert buffer to tensors
        buf_states = torch.tensor(
            np.array(dagger_states), dtype=torch.float32, device=device
        )
        buf_actions = torch.tensor(
            np.array(dagger_actions), dtype=torch.float32, device=device
        )
        # Retrain student on combined original + dagger data
        for ep in tqdm(range(cfg.dagger_epochs), desc="DAgger Train", unit="ep"):
            # sample half batch from original
            s1, *_ = sampler.sample(cfg.distill_batch_size // 2)
            # sample half batch from dagger buffer
            idx = np.random.randint(0, len(buf_states), cfg.distill_batch_size // 2)
            s2 = buf_states[idx]
            a2 = buf_actions[idx]
            states_batch = torch.cat([s1, s2], dim=0)
            with torch.no_grad():
                t1 = teacher.actor.sample(s1)
            targets = torch.cat([t1, a2], dim=0)
            pred = student(states_batch)
            loss = loss_fn(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    # Final save and evaluation
    torch.save(student.state_dict(), cfg.student_model_save_path)
    print(f"Saved student policy to {cfg.student_model_save_path}")

    # Evaluate both
    eval_env = gym.make(teacher_params["gym_env_name"])
    t_r, t_l, t_t = eval_policy(
        teacher, eval_env, seed=cfg.seed, eval_episodes=cfg.eval_episodes
    )
    s_r, s_l, s_t = eval_policy(
        student, eval_env, seed=cfg.seed, eval_episodes=cfg.eval_episodes
    )

    # Save metrics
    rows = []
    for i, (r, l, tm) in enumerate(zip(t_r, t_l, t_t)):
        rows.append(
            {"model": "teacher", "episode": i + 1, "reward": r, "length": l, "time": tm}
        )
    for i, (r, l, tm) in enumerate(zip(s_r, s_l, s_t)):
        rows.append(
            {"model": "student", "episode": i + 1, "reward": r, "length": l, "time": tm}
        )
    pd.DataFrame(rows).to_csv(
        cfg.student_model_save_path.replace(".pt", "_dagger_eval.csv"), index=False
    )


if __name__ == "__main__":
    distill()
