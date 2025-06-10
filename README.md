# Diffusion Q-Learning Dashboard
A Streamlit-based dashboard for visualizing and interacting with a Diffusion-QL policy, built on top of the [Diffusion Policies for Offline RL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL) codebase. This repo contains:

- Upgraded dependencies to modern PyTorch and related libraries
- Hydra configuration replacing the original logging system
- Streamlit dashboard with multiple pages for comprehensive model inspection

## Core Changes

1. **Config and Logging**

2. **Streamlit Dashboard**
    - Home Page
    - Comparative Analysis
    - Policy Visualizer
    - Q-Function Explorer

3. **Performance Optimizations**

## Setup

**Clone repository and install dependencies:**

    ```bash
    git clone https://github.com/RDG0818/Diffusion_Q_Learning_Dashboard.git
    cd Offline-RL-Trajectory-Diffusion-Policy
    conda create -n diff_policy python=3.10
    conda activate diff_policy
    pip install -r requirements.txt
    python setup.py
    ```


## Running

For default run:

```bash
python main.py
```

To modify configs, look at config/config.yaml. To modify specific environments, look under config/env or config/quality.

To run dashboard locally:

``` bash
python -m streamlit run streamlit/app.py
```

## Results

## Acknowledgements

This project builds upon the foundational concepts and codebase from the paper [Diffusion Policies as an Expressive Policy Class For Offline Reinforcement Learning](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL).