import d4rl
import numpy as np
import gym
#TODO: Dynamically combine the datasets
def mix_datasets(first_dataset_name, second_dataset_name) -> dict:
    env = gym.make(first_dataset_name)
    env2 = gym.make(second_dataset_name)
    medium_data = d4rl.qlearning_dataset(env)
    expert_data = d4rl.qlearning_dataset(env2)
    combined_observations = np.concatenate(
        [medium_data["observations"], expert_data["observations"]]
    )
    combined_actions = np.concatenate(
        [medium_data["actions"], expert_data["actions"]]
    )
    combined_next_observations = np.concatenate(
        [medium_data["next_observations"], expert_data["next_observations"]]
    )
    combined_rewards = np.concatenate(
        [medium_data["rewards"], expert_data["rewards"]]
    )
    combined_terminals = np.concatenate(
        [medium_data["terminals"], expert_data["terminals"]]
    )



    medium_source = np.zeros(len(medium_data["observations"]), dtype=int)  # 0 for medium
    expert_source = np.ones(len(expert_data["observations"]), dtype=int)  # 1 for expert
    combined_sources = np.concatenate([medium_source, expert_source])

    combined_data = {
        "observations": combined_observations,
        "actions": combined_actions,
        "next_observations": combined_next_observations,
        "rewards": combined_rewards,
        "terminals": combined_terminals,
        "sources": combined_sources,  # Add source information
    }

    return combined_data

