import d4rl
import numpy as np
import gym
import collections
import random

def get_trajectories(dataset):
    trajectories = []
    current_trajectory = []
    
    n_points = len(dataset['rewards'])
    data_keys = [k for k, v in dataset.items() if isinstance(v, np.ndarray) and len(v) == n_points]

    for i in range(n_points):
        transition = {key: dataset[key][i] for key in data_keys}
        current_trajectory.append(transition)

        if dataset['terminals'][i] or dataset['timeouts'][i]:
            trajectories.append(current_trajectory)
            current_trajectory = []
    return trajectories

def mix_datasets(dataset_configs: list) -> dict:
    all_trajectories_by_source = []
    total_transitions_by_source = []

    for name, ratio in dataset_configs:
        env = gym.make(name)
        dataset = env.get_dataset()
        trajectories = get_trajectories(dataset)
        random.shuffle(trajectories)
        all_trajectories_by_source.append(trajectories)
        total_transitions_by_source.append(sum(len(t) for t in trajectories))
        
    potential_total_transitions = []
    for i, (name, ratio) in enumerate(dataset_configs):
        if ratio > 0:
            potential_total_transitions.append(total_transitions_by_source[i] / ratio)
    
    if not potential_total_transitions:
        return collections.defaultdict(lambda: np.array([]))

    target_total_transitions = int(min(potential_total_transitions))
    
    final_trajectories_with_source = []
    for i, (name, ratio) in enumerate(dataset_configs):
        target_transitions = int(target_total_transitions * ratio)
        
        collected_transitions = 0
        for traj in all_trajectories_by_source[i]:
            if collected_transitions >= target_transitions:
                break
            final_trajectories_with_source.append((traj, i))
            collected_transitions += len(traj)
    
    random.shuffle(final_trajectories_with_source)

    combined_data = collections.defaultdict(list)
    for trajectory, source_id in final_trajectories_with_source:
        for transition in trajectory:
            for key, value in transition.items():
                combined_data[key].append(value)
            combined_data['sources'].append(source_id)
    
    for key, value in combined_data.items():
        combined_data[key] = np.array(value)

    return combined_data

if __name__ == '__main__':
    
    dataset_configs = [
        ('hopper-medium-v2', 0.25),
        ('hopper-expert-v2', 0.75),
    ]

    total_ratio = sum(ratio for name, ratio in dataset_configs)
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, but they sum to {total_ratio}")

    combined_dataset = mix_datasets(dataset_configs)
    
    print("Combining datasets with the following configuration:")
    for name, ratio in dataset_configs:
        print(f"- {name}: {ratio*100:.0f}%")

    print("\nKeys in the new combined dataset:", list(combined_dataset.keys()))
    if 'observations' in combined_dataset and len(combined_dataset['observations']) > 0:
        total_transitions = len(combined_dataset['observations'])
        print("Total number of transitions in combined dataset:", total_transitions)
        
        source_counts = np.bincount(combined_dataset['sources'])
        print("\nTransition counts and percentage per source:")
        for i, (name, ratio) in enumerate(dataset_configs):
            count = source_counts[i] if i < len(source_counts) else 0
            percentage = (count / total_transitions) * 100 if total_transitions > 0 else 0
            print(f"- Source {i} ({name}): {count} transitions ({percentage:.2f}%)")
    else:
        print("\nResulting dataset is empty.")
