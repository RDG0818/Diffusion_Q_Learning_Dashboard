# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0
#
# Heavily refactored for clarity, type hinting, and documentation.

from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from agents.helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
)

class Diffusion(nn.Module):
    """
    A Denoising Diffusion Probabilistic Model (DDPM) implementation that serves
    as the policy for the agent. It learns to generate actions by reversing a
    fixed noising process.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model: nn.Module,
        max_action: float,
        beta_schedule: str = "linear",
        n_timesteps: int = 100,
        predict_epsilon: bool = True,
    ) -> None:
        """
        Initializes the Diffusion policy.

        Args:
            state_dim (int): The dimensionality of the state space.
            action_dim (int): The dimensionality of the action space.
            model (nn.Module): The neural network used for noise prediction.
            max_action (float): The maximum magnitude of any action component.
            beta_schedule (str, optional): The schedule for beta values ('linear',
                                           'cosine', or 'vp'). Defaults to "linear".
            n_timesteps (int, optional): The number of diffusion timesteps. Defaults to 100.
            predict_epsilon (bool, optional): If True, the model predicts the noise;
                                              otherwise, it predicts x0. Defaults to True.
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model
        self.n_timesteps = n_timesteps
        self.predict_epsilon = predict_epsilon

        schedule_fn_map = {
            "linear": linear_beta_schedule,
            "cosine": cosine_beta_schedule,
            "vp": vp_beta_schedule,
        }
        if beta_schedule not in schedule_fn_map:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}")
        betas = schedule_fn_map[beta_schedule](n_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # These are registered as buffers so they are moved to the correct device
        # along with the model, but are not considered model parameters.

        # Buffers for forward process: q(x_t | x_0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Buffers for reverse process: q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),   # Clamped for numerical stability
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # Buffers for deriving x0 from noise
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

    """ Sampling (Reverse Process) """

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Derives the predicted x0 (the original, un-noised sample) from the
        noisy sample x_t and the predicted noise.
        x_0 = (x_t - sqrt(1 - alpha_cumprod_t) * noise_t) / sqrt(alpha_cumprod_t)

        Args:
            x_t (torch.Tensor): The noisy sample at timestep t.
            t (torch.Tensor): The current timestep.
            noise (torch.Tensor): The noise predicted by the model.

        Returns:
            torch.Tensor: The predicted x0.
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the mean and variance of the posterior distribution q(x_{t-1} | x_t, x_0).

        Args:
            x_start (torch.Tensor): The predicted x0.
            x_t (torch.Tensor): The noisy sample at timestep t.
            t (torch.Tensor): The current timestep.

        Returns:
            A tuple containing:
            - (torch.Tensor): The mean of the posterior.
            - (torch.Tensor): The variance of the posterior.
            - (torch.Tensor): The log variance of the posterior (clipped).
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the mean and variance of the reverse process distribution p(x_{t-1} | x_t).

        Args:
            x_t (torch.Tensor): The noisy sample at timestep t.
            t (torch.Tensor): The current timestep.
            state (torch.Tensor): The conditioning state.

        Returns:
            A tuple containing the mean, variance, and log variance.
        """
        predicted_noise = self.model(x_t, t, state)
        x_recon = self.predict_start_from_noise(x_t, t=t, noise=predicted_noise)
        x_recon.clamp_(-self.max_action, self.max_action)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x_t, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a single step of the reverse diffusion process (sampling).

        Args:
            x_t (torch.Tensor): The noisy sample at timestep t.
            t (torch.Tensor): The current timestep.
            state (torch.Tensor): The conditioning state.

        Returns:
            torch.Tensor: The less noisy sample at timestep t-1.
        """
        batch_size = x_t.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x_t=x_t, t=t, state=state)
        noise = torch.randn_like(x_t)
        # No noise is added at the final step (t=0)
        nonzero_mask = (1 - (t == 0).float()).reshape(
            batch_size, *((1,) * (len(x_t.shape) - 1))
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(
        self,
        state: torch.Tensor,
        shape: torch.Size,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Runs the full reverse diffusion process to generate a sample from pure noise.

        Args:
            state (torch.Tensor): The conditioning state.
            shape (torch.Size): The shape of the desired output sample (batch_size, action_dim).
            verbose (bool, optional): If True, shows a tqdm progress bar. Defaults to False.

        Returns:
            torch.Tensor: The final generated sample (action).
        """
        device = self.betas.device
        x_t = torch.randn(shape, device=device)
        
        # Determine the iterable for the loop
        time_range = reversed(range(0, self.n_timesteps))
        if verbose:
            time_range = tqdm(time_range, desc="Diffusion Sampling", leave=False)

        for i in time_range:
            timesteps = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, timesteps, state)

        return x_t

    @torch.no_grad()
    def sample(self, state: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Public-facing method to sample an action, conditioned on a state.
        This method should be called within a `with torch.no_grad():` block
        for efficiency, which is handled by the decorator.

        Args:
            state (torch.Tensor): The conditioning state (batch_size, state_dim).
            verbose (bool, optional): If True, shows a sampling progress bar.

        Returns:
            torch.Tensor: The generated action (batch_size, action_dim).
        """
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, verbose=verbose)
        return action.clamp_(-self.max_action, self.max_action)

    """ Training (Forward Process) """

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs the forward diffusion process: adds noise to the data.
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise

        Args:
            x_start (torch.Tensor): The original, un-noised data (x0).
            t (torch.Tensor): The timesteps to noise to.
            noise (Optional[torch.Tensor], optional): Optional pre-generated noise.
                                                     If None, noise is sampled. Defaults to None.

        Returns:
            torch.Tensor: The noisy sample at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(
        self,
        x_start: torch.Tensor,
        state: torch.Tensor,
        t: torch.Tensor,
        weights: float = 1.0
    ) -> torch.Tensor:
        """
        Calculates the loss for a single batch.

        Args:
            x_start (torch.Tensor): The original, un-noised actions.
            state (torch.Tensor): The conditioning states.
            t (torch.Tensor): The randomly sampled timesteps.
            weights (float, optional): Optional weights for the loss. Defaults to 1.0.

        Returns:
            torch.Tensor: The calculated loss as a single scalar value.
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_output = self.model(x_noisy, t, state)

        target = noise if self.predict_epsilon else x_start
        
        loss = F.mse_loss(predicted_output, target, reduction="none")
        return (loss * weights).mean()

    def loss(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        The main loss function for training the diffusion model.

        Args:
            x (torch.Tensor): A batch of clean actions from the dataset.
            state (torch.Tensor): The corresponding states.

        Returns:
            torch.Tensor: The final loss value for the batch.
        """
        batch_size = x.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t)

    def forward(self, state: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Defines the forward pass of the module. By default, this is set to
        sample an action, making the class behave like a standard policy network.

        Args:
            state (torch.Tensor): The conditioning state.
            verbose (bool, optional): Verbosity for the sampling loop.

        Returns:
            torch.Tensor: The generated action.
        """
        return self.sample(state, verbose=verbose)
