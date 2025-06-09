# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np

class Data_Sampler(object):
    def __init__(self, data, device, reward_tune='no'):
        data_dict = {'observations': np.concatenate([e.observations[:-1] for e in data]),
                     'actions': np.concatenate([e.actions for e in data]),
                     'next_observations': np.concatenate([e.observations[1:] for e in data]),
                     'rewards': np.concatenate([e.rewards for e in data]),
                     'terminals': np.concatenate([e.terminations for e in data])}
        
        data = data_dict

        self.state = torch.from_numpy(data['observations']).float()
        self.action = torch.from_numpy(data['actions']).float()
        self.next_state = torch.from_numpy(data['next_observations']).float()
        reward = torch.from_numpy(data['rewards']).view(-1, 1).float()
        self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float()

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]
        self.device = device

        if reward_tune == 'normalize':
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == 'iql_antmaze':
            reward = reward - 1.0
        elif reward_tune == 'cql_antmaze':
            reward = (reward - 0.5) * 4.0
        self.reward = reward

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))
        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
            self.next_state[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.not_done[ind].to(self.device)
        )


def iql_normalize(reward, not_done):
	trajs_rt = []
	e_return = 0.0
	for i in range(len(reward)):
		e_return += reward[i]
		if not not_done[i]:
			trajs_rt.append(e_return)
			e_return = 0.0
	rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
	reward /= (rt_max - rt_min)
	reward *= 1000.
	return reward
