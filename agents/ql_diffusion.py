# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Diffusion_QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.model2 = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor2 = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model2, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.actor2_lr_scheduler = CosineAnnealingLR(self.actor2_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for _ in range(iterations):
            # if self.step % 1000 == 0: print(f"Epoch: {self.step // 1000}")
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done, source = replay_buffer.sample(batch_size)

            """ Q Training """
            if self.step < 10e4 * 5:
                current_q1, current_q2 = self.critic(state, action)

                if self.max_q_backup:
                    next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                    next_action_rpt = self.ema_model(next_state_rpt)
                    target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                    target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                    target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                    target_q = torch.min(target_q1, target_q2)
                else:
                    next_action = self.ema_model(next_state)
                    target_q1, target_q2 = self.critic_target(next_state, next_action)
                    target_q = torch.min(target_q1, target_q2)

                target_q = (reward + not_done * self.discount * target_q).detach()

                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.grad_norm > 0:
                    critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
                self.critic_optimizer.step()

            """Clustering"""
            with torch.no_grad():
                if self.step < 10e4 * 5: 
                    top_indices = torch.arange(batch_size)
                    bottom_indices = torch.arange(batch_size)
                else:
                    q1, q2 = self.critic(state, action)
                    q_vals = torch.minimum(q1, q2).flatten()
                    # states = state.flatten().to('cpu').numpy()
                    # actions = action.flatten().to('cpu').numpy()
                    # bc_loss_clustering = self.actor.loss(action, state).to('cpu').numpy()
                    # bc_loss2_clustering = self.actor2.loss(action, state).to('cpu').numpy()
                    # features = np.concatenate([states, actions, np.stack([q_vals, bc_loss_clustering, bc_loss2_clustering], axis=-1)], axis=1)
                    # scaler = StandardScaler()
                    # scaled_features = scaler.fit_transform(features)
                    # kmeans = KMeans(n_clusters=2, random_state=0)
                    # cluster_labels = kmeans.predict(scaled_features)

                    _, top_indices = torch.topk(q_vals, int(batch_size*0.9))
                    _, bottom_indices = torch.topk(q_vals, int(batch_size*0.9), largest=False)
            
            """ Policy Training """
            bc_loss = self.actor.loss(action[top_indices], state[top_indices]).mean()
            
            bc_loss2 = self.actor2.loss(action[bottom_indices], state[bottom_indices]).mean()

            new_action = self.actor(state[top_indices])

            q1_new_action, q2_new_action = self.critic(state[top_indices], new_action)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss + self.eta * q_loss
            
            new_action2 = self.actor2(state[bottom_indices])

            q1_new_action2, q2_new_action2 = self.critic(state[bottom_indices], new_action2)
            if np.random.uniform() > 0.5:
                q_loss2 = - q1_new_action2.mean() / q2_new_action2.abs().mean().detach()
            else:
                q_loss2 = - q2_new_action2.mean() / q1_new_action2.abs().mean().detach()
            actor2_loss = bc_loss2 + self.eta * q_loss2 

            self.actor_optimizer.zero_grad()
            self.actor2_optimizer.zero_grad()
            actor_loss.backward()
            actor2_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
                actor2_grad_norms = nn.utils.clip_grad_norm_(self.actor2.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()
            self.actor2_optimizer.step()


            """ Step Target network """
            if self.step % self.update_ema_every == 0 and self.step < 10e4 * 5:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            if self.step < 10e4 * 5: metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value, dim=-1), 1)
        return action[idx].cpu().data.numpy().flatten()
    
    def sample_action_tensor(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value, dim=-1), 1)
        return action[idx].cpu()

    def sample_action_tensor2(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor2.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value, dim=-1), 1)
        return action[idx].cpu()
    
    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))