# Trajectory-Aware Offline RL with Diffusion Models

This project explores advanced techniques for offline reinforcement learning by enhancing a diffusion-based policy (inspired by Diffusion-QL) with a deep understanding of trajectory structure and data quality. The core idea is to move beyond treating offline data as a simple set of i.i.d. transitions and instead leverage temporal context and a probabilistic understanding of data expertise to train a more robust and performant agent.

## Core Changes

1. **Trajectory-Aware Data Handling**
Instead of sampling random, disconnected transitions, this project uses a TrajectorySampler that works with full-length episodes from the D4RL datasets. This enables two features:

-  History-Conditional Policy: The diffusion model is built upon a Gated Recurrent Unit (GRU). This allows the policy to be conditioned on a sequence of past states, giving it a memory of recent events to make more causally-sound and temporally-consistent decisions.

- N-Step Critic Updates: The critic is trained using N-step returns calculated over the sampled sequences. This provides a more stable and lower-variance training signal compared to standard 1-step TD-learning, leading to a more accurate Q-function.

## Requirements

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RDG0818/Offline-RL-Trajectory-Diffusion-Policy.git
    cd Offline-RL-Trajectory-Diffusion-Policy
    conda create -n diff_policy python=3.9
    conda activate diff_policy
    ```

2.  **Install Dependencies:**
    Installations of [PyTorch](https://pytorch.org/get-started/locally/), [MuJoCo](https://github.com/google-deepmind/mujoco), and [D4RL](https://github.com/Farama-Foundation/D4RL) are required. Additionally, run the following command to install other dependencies.
    ```bash
    pip install -r requirements.txt
    ```


## Running

## Results

## Acknowledgements

This project builds upon the foundational concepts and codebase from the paper [Diffusion Policies as an Expressive Policy Class For Offline Reinforcement Learning](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL).