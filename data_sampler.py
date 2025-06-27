# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0
#
# Heavily refactored for clarity, type hinting, and documentation.

import torch
import numpy as np
from typing import Tuple, List, Union
import minari


class Data_Sampler(object):
    """
    A data sampler for offline reinforcement learning datasets.

    This class processes a Minari dataset into PyTorch tensors and provides a
    method to sample minibatches for training. The entire dataset is loaded
    into CPU memory for fast indexing. Batches are moved to the target device
    on demand.

    Attributes:
        size (int): The total number of transitions in the dataset.
        state_dim (int): The dimensionality of the state space.
        action_dim (int): The dimensionality of the action space.
    """
    def __init__(
        self,
        data: List[minari.EpisodeData],
        device: Union[str, torch.device],
        reward_tune: str = "no"
    ) -> None:
        
        first_episode_obs = data[0].observations

        if isinstance(first_episode_obs, dict):
            data_dict = {
                "observations": np.concatenate([e.observations['observation'][:-1] for e in data]),
                "actions": np.concatenate([e.actions for e in data]),
                "next_observations": np.concatenate([e.observations['observation'][1:] for e in data]),
                "rewards": np.concatenate([e.rewards for e in data]),
                "terminals": np.concatenate([e.terminations for e in data]),
            }
        else:   
            data_dict = {
                "observations": np.concatenate([e.observations[:-1] for e in data]),
                "actions": np.concatenate([e.actions for e in data]),
                "next_observations": np.concatenate([e.observations[1:] for e in data]),
                "rewards": np.concatenate([e.rewards for e in data]),
                "terminals": np.concatenate([e.terminations for e in data]),
            }

        self.state = torch.from_numpy(data_dict["observations"]).float()
        self.action = torch.from_numpy(data_dict["actions"]).float()
        self.next_state = torch.from_numpy(data_dict["next_observations"]).float()
        reward = torch.from_numpy(data_dict["rewards"]).view(-1, 1).float()
        self.not_done = 1.0 - torch.from_numpy(data_dict["terminals"]).view(-1, 1).float()

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]
        self.device = device

        if reward_tune == "normalize":
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == "iql_antmaze":
            reward = reward - 1.0
        elif reward_tune == "cql_antmaze":
            reward = (reward - 0.5) * 4.0
        self.reward = reward

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Samples a random minibatch of transitions from the dataset.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            A tuple containing the batch of transitions:
            (state, action, next_state, reward, not_done)
            All tensors are on the target device.
        """
        ind = torch.randint(0, self.size, size=(batch_size,))
        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
            self.next_state[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.not_done[ind].to(self.device),
        )
