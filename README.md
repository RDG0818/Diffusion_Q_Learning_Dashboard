# Diffusion Q-Learning Dashboard
A Streamlit-based dashboard for visualizing and interacting with a Diffusion-QL policy, built on top of the [Diffusion Policies for Offline RL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL) codebase. This repo contains:

- Upgraded dependencies to modern PyTorch and related libraries
- Hydra configuration replacing the original logging system
- Streamlit dashboard with multiple pages for comprehensive model inspection

## Core Changes

1. **Config and Logging**: Switched to Hydra for flexible experiment configuration and logging

2. **Streamlit Dashboard**
    - Home Page: Overview of Diffusion-QL and reproduction of results
    - Comparative Analysis: Line plots of reward & Q-learning loss over time
    - Policy Visualizer: In-browser video of the agent’s rollout and total reward
    - Q-Function Explorer: 2D heatmaps of Q(s, a) across action-dimension pairs and violin plots of Q-value distributions

3. **Updated Libraries**: Migrated to PyTorch ≥ 2.0 for faster inference and Minari for modern datasets

4. **DAgger-based Policy Distillation**: Introduced a DAgger-driven `distill.py` workflow to train a lightweight MLP student policy that imitates the Diffusion-QL teacher

## Setup

**Clone repository and install dependencies:**

Note that you should have at least 5 gb of spare memory for the Minari datasets.

    ```
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

After creating a results directory from running main.py, you can generate a distilled MLP by setting the results folder in config/distill.yaml and running:

```
python distill.py
```

To run dashboard locally:

``` bash
python -m streamlit run streamlit/app.py
```

## Results
![Alt text](dashboard_images/st1.png "Home Page")

![Alt text](dashboard_images/st2.png "Home Page")

![Alt text](dashboard_images/st3.png "Q-function Explorer")

![Alt text](dashboard_images/st4.png "Policy Visualizer")

## Acknowledgements

This project builds upon the foundational concepts and codebase from the paper [Diffusion Policies as an Expressive Policy Class For Offline Reinforcement Learning](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL).