# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0
#
# Heavily refactored for clarity, type hinting, and documentation.

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Union, Dict, List, Optional, Any
from tqdm.auto import tqdm

from agents.diffusion import Diffusion
from agents.model import MLP, Critic
from agents.helpers import EMA
from data_sampler import Data_Sampler
import os

@dataclass
class AgentConfig:
    """Configuration object for the Diffusion_QL agent's hyperparameters."""
    discount: float 
    tau: float
    eta: float 
    lr: float 
    lr_decay: bool
    lr_maxt: int
    grad_norm: float
    beta_schedule: str 
    n_timesteps: int 
    max_q_backup: bool
    ema_decay: float
    step_start_ema: int
    update_ema_every: int 

class Diffusion_QL(object):
    """
    Main agent class for Diffusion Q-Learning.

    This agent combines a diffusion model for policy representation with a
    TD3-style twin critic for Q-function learning. The policy is trained
    with a combination of a behavior cloning loss and a Q-learning loss.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: Union[str, torch.device],
        cfg: AgentConfig,
    ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.actor = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=self.model,
            max_action=max_action,
            beta_schedule=cfg.beta_schedule,
            n_timesteps=cfg.n_timesteps,
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.ema = EMA(cfg.ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.step = 0
        self.step_start_ema = cfg.step_start_ema
        self.update_ema_every = cfg.update_ema_every
        self.tau = cfg.tau

        self.lr_decay = cfg.lr_decay
        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer, T_max=cfg.lr_maxt, eta_min=0.0
            )
            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer, T_max=cfg.lr_maxt, eta_min=0.0
            )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = cfg.discount
        self.eta = cfg.eta  # q_learning weight
        self.grad_norm = cfg.grad_norm
        self.max_q_backup = cfg.max_q_backup
        self.device = device

    def step_ema(self) -> None:
        """Updates the EMA weights of the actor model."""
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(
        self,
        replay_buffer: Data_Sampler,
        iterations: int,
        batch_size: int = 256,
        progress_bar: Optional[tqdm] = None,
    ) -> Dict[str, List[float]]:
        """
        Performs a fixed number of training steps.

        Args:
            replay_buffer (Data_Sampler): The buffer to sample training data from.
            iterations (int): The number of training steps to perform.
            batch_size (int, optional): The size of each training batch. Defaults to 256.
            progress_bar (Optional[tqdm], optional): A tqdm progress bar to update. Defaults to None.

        Returns:
            Dict[str, List[float]]: A dictionary containing lists of loss values and other metrics for the training run.
        """

        metric = {
            "bc_loss": [],
            "ql_loss": [],
            "actor_loss": [],
            "critic_loss": [],
            "avg_q_batch": [],
            "avg_q_policy": [],
            "actor_gn": [],
            "critic_gn": [],
        }
        for _ in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(
                batch_size
            )

            """ Q Training """

            with torch.no_grad():
                if self.max_q_backup:
                    next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                    next_action_rpt = self.ema_model(next_state_rpt)
                    target_q1, target_q2 = self.critic_target(
                        next_state_rpt, next_action_rpt
                    )
                    target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                    target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                    target_q = torch.min(target_q1, target_q2)
                else:
                    next_action = self.ema_model(next_state)
                    target_q1, target_q2 = self.critic_target(next_state, next_action)
                    target_q = torch.min(target_q1, target_q2)

                target_q = (reward + not_done * self.discount * target_q).detach()

            current_q1, current_q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=self.grad_norm, norm_type=2
                )
            self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.actor.loss(action, state)
            new_action = self.actor(state)

            q1_new_action, q2_new_action = self.critic(state, new_action)
            if np.random.uniform() > 0.5:
                q_loss = -q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = -q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=self.grad_norm, norm_type=2
                )
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Logging """
            metric["actor_loss"].append(actor_loss.item())
            metric["bc_loss"].append(bc_loss.item())
            metric["ql_loss"].append(q_loss.item())
            metric["critic_loss"].append(critic_loss.item())
            metric["avg_q_batch"].append(torch.min(current_q1, current_q2).mean().item())
            metric["avg_q_policy"].append(torch.min(q1_new_action, q2_new_action).mean().item())
            if self.grad_norm > 0:
                metric["actor_gn"].append(actor_grad_norms.item())
                metric["critic_gn"].append(critic_grad_norms.item())

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    critic_loss=f"{metric['critic_loss'][-1]:.3f}",
                    actor_loss=f"{metric['actor_loss'][-1]:.3f}",
                )

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    @torch.no_grad()
    def sample_action(self, state: np.ndarray) -> np.ndarray:
        """
        Samples an action from the policy for a given state.

        This method generates a batch of candidate actions using the diffusion
        model, evaluates them with the critic, and returns the action with
        the highest Q-value according to a softmax distribution.

        Args:
            state (np.ndarray): The input state, with shape (state_dim,).

        Returns:
            np.ndarray: The sampled action, with shape (action_dim,).
        """
        state_tensor = torch.from_numpy(state).float().to(self.device).reshape(1, -1)
        state_rpt = torch.repeat_interleave(state_tensor, repeats=50, dim=0)

        action_candidates = self.ema_model.sample(state_rpt)
        q_values = self.critic_target.q_min(state_rpt, action_candidates).flatten()

        idx = torch.multinomial(F.softmax(q_values, dim=-1), num_samples=1)
        
        return action_candidates[idx].cpu().numpy().flatten()

    def save_model(self, dir_path: str, id_num: Optional[int] = None) -> None:
        """
        Saves the state dictionaries of the actor and critic networks.

        Args:
            dir_path (str): The directory where models will be saved.
            id_num (Optional[int], optional): An optional identifier (like epoch number)
                                              to append to the filename. Defaults to None.
        """
        if id_num is not None:
            torch.save(self.actor.state_dict(), os.path.join(dir_path, f"actor_{id_num}.pth"))
            torch.save(self.critic.state_dict(), os.path.join(dir_path, f"critic_{id_num}.pth"))
        else:
            torch.save(self.actor.state_dict(), os.path.join(dir_path, "actor.pth"))
            torch.save(self.critic.state_dict(), os.path.join(dir_path, "critic.pth"))

    def load_model(self, dir_path: str, id_num: Optional[int] = None) -> None:
        """
        Loads the state dictionaries for the actor and critic networks.

        This method includes logic to strip the '_orig_mod.' prefix that
        `torch.compile` may add to state dictionary keys.

        Args:
            dir_path (str): The directory from which to load models.
            id_num (Optional[int], optional): The identifier of the models to load.
                                              Defaults to None.
        """
        if id_num is not None:
            actor_path = os.path.join(dir_path, f"actor_{id_num}.pth")
            critic_path = os.path.join(dir_path, f"critic_{id_num}.pth")
        else:
            actor_path = os.path.join(dir_path, "actor.pth")
            critic_path = os.path.join(dir_path, "critic.pth")

        def strip_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Removes the '_orig_mod.' prefix from torch.compile."""
            return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        actor_state_dict = torch.load(actor_path, map_location=self.device)
        self.actor.load_state_dict(strip_prefix(actor_state_dict))

        critic_state_dict = torch.load(critic_path, map_location=self.device)
        self.critic.load_state_dict(strip_prefix(critic_state_dict))

        self.critic_target = copy.deepcopy(self.critic)
        self.ema_model = copy.deepcopy(self.actor)
