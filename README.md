# Updated Diffusion Q Learning
This repository provides a modern implementation of the paper [Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/pdf/2208.06193). The goal of this project is to provide an easy-to-use and extensible benchmark for further research in offline reinforcment learning with diffusion models.

## Core Changes

- Natively handles `hopper`, `walker2d`, `halfcheetah`, `antmaze`, `kitchen`, and `pen` environments from the **Minari** benchmarks.

- All hyperparameters and settings are manage with **Hydra**, allowing for swappable configurations via the command line or config.yaml.

- All logging is done through **Weights & Biases** for automatic visualization, analysis, and hyperparameter tuning.

- Codebase is fully type-hinted and documented, and it also includes a `evaluation.py` script for performance tests on model checkpoints.

## Installation

### Clone the Repository:

```bash
git clone https://github.com/RDG0818/Updated_Diffusion_Q_Learning.git
cd Updated_Diffusion_Q_Learning
```

### Create Conda Environment and Install Dependencies:
```bash
conda create -n diff_policy python=3.10
conda activate diff_policy
pip install -r requirements.txt
pip install -e .
```

### Download Datasets:
Run the setup script to download the necessary Minari datasets. Note that this may take some time and requires at least 5-10 GB of free disk space.
```bash
python dataset_download.py
```

## Usage

### Training a Model

To start a training run, execute `main.py`. You can override any configuration setting from the command line. 

```bash
python main.py
```
**Train on different environment:**

```bash 
python main.py env=walker2d_medium_expert

python main.py env=pen_human agent.lr=0.00003 agent.eta=0.15 reward_tune="normalize" agent.grad_norm=1.0 eval_episodes=50
```

### Evaluating a Trained Model

Use the `evaluation.py` script to load a savedcheckpoint from a completed run and evaluate its performance.

```bash
python evaluation.py --run_dir results/hopper_medium_expert-0
```

To change the environment or modify the hyperparameters, look at config/config.yaml.

## Results

Below are the training curves from W&B.

![alt text](https://github.com/RDG0818/Updated_Diffusion_Q_Learning/images/wandb_image.png "Diffusion Q Learning")


## Citation

If you use this code in your research, please consider citing it:

```bibtex
@software{Goodwin_2025_DiffusionQL,
  author = {[Ryan] [Goodwin]},
  title = {{A Refactored Implementation of Diffusion Policies for Offline RL}},
  month = {6},
  year = {2025},
  url = {https://github.com/RDG0818/Updated_Diffusion_Q_Learning}
}
```


## Acknowledgements

This project is a refactoring and extension of the original work by Zhendong Wang, et al. from the following paper and codebase: [Diffusion Policies as an Expressive Policy Class For Offline Reinforcement Learning](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL).

## License

This project is licensed under the **Apache 2.0 License**.