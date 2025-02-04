# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix

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

# class Cluster:
#     def __init__(self, n_clusters=2, batch_size=256, n_init=10):
#         self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, n_init=n_init)
#         self.scalar = StandardScaler()
#         self.cluster_centers = None

#     def features(self, state, action, next_state, q_value, bc_loss, bc_loss2):
#         state = state.flatten().to('cpu').numpy().reshape(state.shape[0], -1)
#         action = action.flatten().to('cpu').numpy().reshape(action.shape[0], -1)
#         next_state = next_state.flatten().to('cpu').numpy().reshape(action.shape[0], -1)
#         q_value = q_value.reshape(-1, 1).to('cpu').numpy()
#         bc_loss = bc_loss.to('cpu').numpy()
#         bc_loss2 = bc_loss2.to('cpu').numpy()
#         return np.concatenate([state, action, next_state, q_value, bc_loss, bc_loss2], axis = 1)

#     def predict(self, features):
#         if self.cluster_centers is None: # check if cluster centers are initialized
#             self.scalar.fit(features)
#             scaled_features = self.scalar.transform(features)
#             self.cluster_centers = self.kmeans.fit(scaled_features).cluster_centers_ # fit on first batch
#         else:
#             scaled_features = self.scalar.transform(features)
#         prediction = self.kmeans.predict(scaled_features)
#         self.kmeans = self.kmeans.partial_fit(scaled_features)
#         return prediction


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
                 cluster=None
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.model2 = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.model3 = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor2 = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model2, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor3 = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model3, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=lr)
        self.actor3_optimizer = torch.optim.Adam(self.actor3.parameters(),  lr=lr)


        self.cluster = cluster
        self.critic_training_time = 10e4 * 5
        self.n_clusters = self.cluster.n_clusters if self.cluster is not None else 0

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
            self.actor3_lr_scheduler = CosineAnnealingLR(self.actor3_optimizer, T_max=lr_maxt, eta_min=0.)
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
            state, action, next_state, reward, not_done, source = replay_buffer.sample(batch_size)
            state_copy = state.cpu().numpy().astype(np.float64)
            if self.cluster is not None:
                labels = torch.from_numpy(self.cluster.predict(state_copy)).to(self.device)

            """ Q Training """
            if self.step < self.critic_training_time:
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
                if self.step < self.critic_training_time: 
                    top_indices = torch.arange(batch_size)
                    bottom_indices = torch.arange(batch_size)
                else:
                    expert_estimate = torch.zeros(batch_size, dtype=torch.bool)
                    non_expert_estimate = torch.zeros(batch_size, dtype=torch.bool)
                    estimate = torch.zeros(batch_size, dtype=torch.bool)
                    indices = torch.arange(batch_size).to(self.device)
                    q1, q2 = self.critic(state, action)
                    q_vals = torch.minimum(q1, q2).flatten()
                    
                    if self.cluster is not None:
                        for i in range(self.n_clusters):
                            cluster_indices = indices[labels == i]

                            if cluster_indices.numel() > 0:  # Check for empty cluster   
                                q_mean = q_vals[cluster_indices].mean()
                                cluster_size = cluster_indices.shape[0]
                                q_indices = cluster_indices[q_vals[cluster_indices] > q_mean]
                                _, cluster_q_indices = torch.topk(q_vals[cluster_indices], int(cluster_size * .5))
                                _, cluster_q_bottom_indices = torch.topk(q_vals[cluster_indices], int(cluster_size * .5), largest=False)
                                expert_indices = cluster_indices[cluster_q_indices]  # Directly filter cluster indices
                                non_expert_indices = cluster_indices[cluster_q_bottom_indices]
                                expert_estimate[expert_indices] = True
                                non_expert_estimate[non_expert_indices] = True
                                estimate[q_indices] = True

                        top_indices = indices[expert_estimate]
                        bottom_indices = indices[non_expert_estimate]
                    else:
                        q_mean = q_vals.mean()
                        top_indices = indices[q_vals > q_mean]
                        bottom_indices = indices[q_vals < q_mean]
                    
                    if self.step % 10000 == 0: 
                        print("Time Step:", self.step)
                        temp = q_vals > q_mean if self.cluster is None else estimate
                        temp = temp.cpu().numpy().astype(np.int32)
                        cm = confusion_matrix(source.cpu(), temp)
                        print(cm)


            """ Policy Training """

            if self.step < self.critic_training_time:
                bc_loss = self.actor.loss(action, state).mean()
                new_action = self.actor(state)
                q1_new_action, q2_new_action = self.critic(state, new_action)
                if np.random.uniform() > 0.5:
                    q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
                else:
                    q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
                actor_loss = bc_loss + self.eta * q_loss
                if self.grad_norm > 0:
                    actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
            else:
                bc_loss2 = self.actor2.loss(action[top_indices], state[top_indices]).mean()
                bc_loss3 = self.actor3.loss(action[bottom_indices], state[bottom_indices]).mean()
                
                new_action2 = self.actor2(state[top_indices])

                q1_new_action2, q2_new_action2 = self.critic(state[top_indices], new_action2)
                if np.random.uniform() > 0.5:
                    q_loss2 = - q1_new_action2.mean() / q2_new_action2.abs().mean().detach()
                else:
                    q_loss2 = - q2_new_action2.mean() / q1_new_action2.abs().mean().detach()
                actor2_loss = bc_loss2 + self.eta * q_loss2 

                new_action3 = self.actor3(state[bottom_indices])

                q1_new_action3, q2_new_action3 = self.critic(state[bottom_indices], new_action3)
                if np.random.uniform() > 0.5:
                    q_loss3 = - q1_new_action3.mean() / q2_new_action3.abs().mean().detach()
                else:
                    q_loss3 = - q2_new_action3.mean() / q1_new_action3.abs().mean().detach()
                actor3_loss = bc_loss3 - self.eta * q_loss3

                if self.grad_norm > 0: 
                    actor2_grad_norms = nn.utils.clip_grad_norm_(self.actor2.parameters(), max_norm=self.grad_norm, norm_type=2)
                
                self.actor2_optimizer.zero_grad()
                self.actor3_optimizer.zero_grad()
                actor2_loss.backward()
                actor3_loss.backward()
                self.actor2_optimizer.step()
                self.actor3_optimizer.step()
            
            """ Step Target network """
            if self.step % self.update_ema_every == 0 and self.step < self.critic_training_time:
                self.step_ema()

            if self.step <self.critic_training_time: 
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            
            if self.step < self.critic_training_time: metric['actor_loss'].append(actor_loss.item())
            if self.step < self.critic_training_time: metric['bc_loss'].append(bc_loss.item())
            if self.step < self.critic_training_time: metric['ql_loss'].append(q_loss.item())
            if self.step < self.critic_training_time: metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt) if self.step < self.critic_training_time else self.actor2.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value, dim=-1), 1)
        return action[idx].cpu().data.numpy().flatten()
    
    def sample_action_tensor(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor2.sample(state_rpt)
            q_value = self.critic.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value, dim=-1), 1)
        return action[idx].cpu()

    def sample_action_tensor2(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor3.sample(state_rpt)
            q_value = self.critic.q_min(state_rpt, action).flatten()
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