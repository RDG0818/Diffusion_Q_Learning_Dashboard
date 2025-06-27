# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0
#
# Heavily refactored for clarity, type hinting, and documentation.

import torch
import torch.nn as nn
from typing import Tuple, Union

from agents.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron used as the noise prediction network in a
    diffusion model.

    This network takes a noisy action, a timestep, and a state as input,
    and it outputs the predicted noise that was added to the action.

    Attributes:
        time_mlp (nn.Sequential): A sub-network to create time embeddings.
        mid_layer (nn.Sequential): The main processing block of the MLP.
        final_layer (nn.Linear): The output layer that predicts the noise.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: Union[str, torch.device],
        t_dim: int = 16
    ) -> None:
        """
        Initializes the MLP noise prediction network.

        Args:
            state_dim (int): The dimensionality of the state space.
            action_dim (int): The dimensionality of the action space.
            device (Union[str, torch.device]): The device to run the model on.
            t_dim (int, optional): The dimensionality of the time embedding.
                                   Defaults to 16.
        """
        super().__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
        )

        self.final_layer = nn.Linear(256, action_dim)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the forward pass of the noise prediction network.

        Args:
            x (torch.Tensor): The noisy action tensor (batch_size, action_dim).
            time (torch.Tensor): The timestep tensor (batch_size,).
            state (torch.Tensor): The state tensor (batch_size, state_dim).

        Returns:
            torch.Tensor: The predicted noise tensor (batch_size, action_dim).
        """
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)


class Critic(nn.Module):
    """
    A Twin-Critic network implementation for Q-value estimation, designed to mitigate overestimation bias in Q-learning.

    Attributes:
        q1_model (nn.Sequential): The first Q-value network.
        q2_model (nn.Sequential): The second Q-value network.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ) -> None:
        """
        Initializes the Critic network.

        Args:
            state_dim (int): The dimensionality of the state space.
            action_dim (int): The dimensionality of the action space.
            hidden_dim (int, optional): The size of the hidden layers. Defaults to 256.
        """
        super().__init__()
        # First Q-network
        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

        # Second Q-network
        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Computes the element-wise minimum of the two Q-networks.

        Args:
            state (torch.Tensor): The batch of states.
            action (torch.Tensor): The batch of actions.

        Returns:
            torch.Tensor: The element-wise minimum of the two Q-value estimates.
        """
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)