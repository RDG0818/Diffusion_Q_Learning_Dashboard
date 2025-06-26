# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, Type, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

""" Time/Positional Embedding """

class SinusoidalPosEmb(nn.Module):
    """
    A module for creating sinusoidal positional embeddings, used for
    encoding timesteps in the diffusion model.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

""" Scheduling """

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extracts values from tensor 'a' at the indices specified by 't' and
    reshapes them to be broadcastable with a tensor of shape 'x_shape'.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generates a cosine tau schedule for the diffusion process, as proposed in:
    https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generates a linear tau schedule for the diffusion process.
    """
    betas = np.linspace(beta_start, beta_end, timesteps)
    return torch.tensor(betas, dtype=dtype)


def vp_beta_schedule(timesteps, dtype=torch.float32):
    """    
    Generates a variance-preserving (VP) tau schedule, as described in
    "Score-Based Generative Modeling through Stochastic Differential Equations".
    """
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)


class EMA:
    """
    A helper class for maintaining an Exponential Moving Average of a model's weights.
    """

    def __init__(self, tau: float) -> None:
        super().__init__()
        self.tau = tau

    def update_model_average(self,ma_model: nn.Module, current_model: nn.Module) -> None:
        """
        Updates the weights of the moving average model (ma_model) using the
        weights from the current model.

        Args:
            ma_model (nn.Module): The target model to be updated (e.g., actor_target).
            current_model (nn.Module): The source model with the latest weights.
        """
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
        """
        Calculates the new moving average value for a single parameter tensor.

        Args:
            old (Optional[torch.Tensor]): The old moving average weight.
            new (torch.Tensor): The new weight from the online network.

        Returns:
            torch.Tensor: The updated moving average weight.
        """
        if old is None:
            return new
        return old * self.tau + (1 - self.tau) * new

