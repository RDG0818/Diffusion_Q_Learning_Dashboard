# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import time
import math
import torch
import numpy as np

class Data_Sampler(object):
	def __init__(self, data, device, reward_tune='no', mixed_data=False):
		
		self.state = torch.from_numpy(data['observations']).float()
		self.action = torch.from_numpy(data['actions']).float()
		self.next_state = torch.from_numpy(data['next_observations']).float()
		reward = torch.from_numpy(data['rewards']).view(-1, 1).float()
		self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float()
		self.mixed_data = mixed_data
		if self.mixed_data:	
			self.sources = torch.from_numpy(data['sources'])

		self.size = self.state.shape[0]
		self.state_dim = self.state.shape[1]
		self.action_dim = self.action.shape[1]

		self.device = device

		if reward_tune == 'normalize':
			reward = (reward - reward.mean()) / reward.std()
		elif reward_tune == 'iql_antmaze':
			reward = reward - 1.0
		elif reward_tune == 'iql_locomotion':
			reward = iql_normalize(reward, self.not_done)
		elif reward_tune == 'cql_antmaze':
			reward = (reward - 0.5) * 4.0
		elif reward_tune == 'antmaze':
			reward = (reward - 0.25) * 2.0
		self.reward = reward

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, size=(batch_size,))
		if self.mixed_data:
			return (
				self.state[ind].to(self.device),
				self.action[ind].to(self.device),
				self.next_state[ind].to(self.device),
				self.reward[ind].to(self.device),
				self.not_done[ind].to(self.device),
				self.sources[ind].to(self.device)
			)
		return (
			self.state[ind].to(self.device),
			self.action[ind].to(self.device),
			self.next_state[ind].to(self.device),
			self.reward[ind].to(self.device),
			self.not_done[ind].to(self.device)
		)


def iql_normalize(reward, not_done):
	trajs_rt = []
	episode_return = 0.0
	for i in range(len(reward)):
		episode_return += reward[i]
		if not not_done[i]:
			trajs_rt.append(episode_return)
			episode_return = 0.0
	rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
	reward /= (rt_max - rt_min)
	reward *= 1000.
	return reward


import torch
import numpy as np
import collections
from typing import Dict, List, Any, Tuple

class TrajectorySampler(object):
    """
    A data sampler that processes a flat D4RL-style dataset into trajectories
    and samples sequences of a fixed length from them.

    This sampler is designed to provide chunks of sequential data, which is useful
    for training models that can leverage temporal context, such as Transformers
    or other sequence models. It filters out trajectories shorter than the
    specified sequence length to avoid the need for padding.
    """
    def __init__(self, data: Dict[str, np.ndarray], device: torch.device, sequence_length: int, reward_tune: str = 'no') -> None:
        """
        Initializes the TrajectorySampler.

        Args:
            data (Dict[str, np.ndarray]): A dictionary containing the dataset,
                expected to have keys like 'observations', 'actions', etc.
            device (torch.device): The device (CPU or GPU) to move the sampled
                tensors to.
            sequence_length (int): The fixed length of the sequences to sample.
            reward_tune (str, optional): The type of reward tuning to apply.
                Currently supports 'normalize'. Defaults to 'no'.
        """
        self.device: torch.device = device
        self.sequence_length: int = sequence_length
        self.mixed_data: bool = 'sources' in data
        self.trajectories: List[Dict[str, np.ndarray]] = []
        
        if reward_tune == 'normalize':
            data['rewards'] = (data['rewards'] - data['rewards'].mean()) / data['rewards'].std()
        
        self._process_and_filter_trajectories(data)

        if not self.trajectories:
            raise ValueError(
                f"No trajectories are long enough for the given sequence_length of {sequence_length}."
            )
            
        self.state_dim: int = self.trajectories[0]['observations'].shape[1]
        self.action_dim: int = self.trajectories[0]['actions'].shape[1]
        
        self._precompute_indices()

    def _process_and_filter_trajectories(self, data: Dict[str, np.ndarray]) -> None:
        """
        Parses the flat dataset into trajectories and filters out those
        that are shorter than the required sequence length.

        Args:
            data (Dict[str, np.ndarray]): The raw dataset dictionary.
        """
        current_traj: List[Dict[str, Any]] = []
        n_points: int = len(data['rewards'])
        data_keys: List[str] = [k for k, v in data.items() if isinstance(v, np.ndarray) and len(v) == n_points]

        for i in range(n_points):
            current_traj.append({key: data[key][i] for key in data_keys})

            if data['terminals'][i] or data['timeouts'][i]:
                if len(current_traj) >= self.sequence_length:
                    traj_dict = collections.defaultdict(list)
                    for transition in current_traj:
                        for key, value in transition.items():
                            traj_dict[key].append(value)
                    
                    traj_numpy = {k: np.array(v) for k, v in traj_dict.items()}
                    traj_numpy['not_done'] = 1. - traj_numpy['terminals'].reshape(-1, 1)
                    del traj_numpy['terminals']
                    if 'timeouts' in traj_numpy:
                        del traj_numpy['timeouts']
                    
                    self.trajectories.append(traj_numpy)
                current_traj = []

    def _precompute_indices(self) -> None:
        """
        Pre-computes indices to allow for efficient sampling of valid sequences.
        This method calculates the number of possible starting points for a
        sequence in each trajectory and their cumulative sum.
        """
        self.valid_start_points: np.ndarray = np.array(
            [len(traj['rewards']) - self.sequence_length + 1 for traj in self.trajectories]
        )
        self.cumulative_start_points: np.ndarray = np.cumsum(self.valid_start_points)
        self.total_start_points: int = self.cumulative_start_points[-1]

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Samples a batch of fixed-length sequences from the dataset.

        Args:
            batch_size (int): The number of sequences to sample.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple containing the batch of sequences.
                The order is (observations, actions, next_observations, rewards,
                not_done) and optionally 'sources' if the data is mixed. Each
                element is a tensor of shape (batch_size, sequence_length, ...).
        """
        start_indices_global: np.ndarray = np.random.randint(0, self.total_start_points, size=batch_size)
        traj_indices: np.ndarray = np.searchsorted(self.cumulative_start_points, start_indices_global, side='right')
        
        prior_cumulative: np.ndarray = np.concatenate(([0], self.cumulative_start_points[:-1]))
        indices_in_traj: np.ndarray = start_indices_global - prior_cumulative[traj_indices]
        
        batch = collections.defaultdict(list)

        for i in range(batch_size):
            traj_idx: int = traj_indices[i]
            start_idx: int = indices_in_traj[i]
            end_idx: int = start_idx + self.sequence_length
            
            selected_traj: Dict[str, np.ndarray] = self.trajectories[traj_idx]

            for key, values in selected_traj.items():
                sequence = values[start_idx:end_idx]
                batch[key].append(sequence)

        tensors: Dict[str, torch.Tensor] = {}
        for key, value_list in batch.items():
            if key == 'sources':
                tensors[key] = torch.from_numpy(np.stack(value_list)).long().to(self.device)
            else:
                tensors[key] = torch.from_numpy(np.stack(value_list)).float().to(self.device)

        if self.mixed_data:
            return (
                tensors['observations'],
                tensors['actions'],
                tensors['next_observations'],
                tensors['rewards'],
                tensors['not_done'],
                tensors['sources'],
            )
        else:
            return (
                tensors['observations'],
                tensors['actions'],
                tensors['next_observations'],
                tensors['rewards'],
                tensors['not_done'],
            )

