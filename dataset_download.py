import minari
from typing import Dict, Any, List, Set

DATASET_CONFIG: List[Dict[str, Any]] = [
    {
        "type": "combine",
        "target_id": "mujoco/halfcheetah/medium-expert-v0",
        "source_ids": ["mujoco/halfcheetah/medium-v0", "mujoco/halfcheetah/expert-v0"],
    },
    {
        "type": "combine",
        "target_id": "mujoco/hopper/medium-expert-v0",
        "source_ids": ["mujoco/hopper/medium-v0", "mujoco/hopper/expert-v0"],
    },
    {
        "type": "combine",
        "target_id": "mujoco/walker2d/medium-expert-v0",
        "source_ids": ["mujoco/walker2d/medium-v0", "mujoco/walker2d/expert-v0"],
    },
    {
        "type": "download",
        "target_id": "D4RL/antmaze/umaze-v1",
    },
    {
        "type": "download",
        "target_id": "D4RL/antmaze/umaze-diverse-v1",
    },
    {
        "type": "download",
        "target_id": "D4RL/antmaze/medium-play-v1",
    },
    {
        "type": "download",
        "target_id": "D4RL/antmaze/medium-diverse-v1",
    },
    {
        "type": "download",
        "target_id": "D4RL/antmaze/large-play-v1",
    },
    {
        "type": "download",
        "target_id": "D4RL/antmaze/large-diverse-v1",
    },
    {
        "type": "download",
        "target_id": "D4RL/pen/human-v2",
    },
    {
        "type": "download",
        "target_id": "D4RL/pen/cloned-v2",
    },
    {
        "type": "download",
        "target_id": "D4RL/kitchen/complete-v2",
    },
    {
        "type": "download",
        "target_id": "D4RL/kitchen/partial-v2",
    },
    {
        "type": "download",
        "target_id": "D4RL/kitchen/mixed-v2",
    }
]


def download_if_missing(dataset_id: str, local_datasets: Set[str]) -> bool:
    """
    Downloads a Minari dataset if it doesn't already exist locally.

    Args:
        dataset_id (str): The ID of the dataset to download.
        local_datasets (Set[str]): A set of locally available dataset IDs.

    Returns:
        bool: True if the dataset exists locally or was downloaded successfully,
              False otherwise.
    """
    if dataset_id in local_datasets:
        print(f"--> Source '{dataset_id}' already exists locally. Skipping download.")
        return True
    else:
        try:
            print(f"--> Downloading source '{dataset_id}'...")
            minari.download_dataset(dataset_id=dataset_id)
            print(f"--> Successfully downloaded '{dataset_id}'.")
            return True
        except Exception as e:
            print(
                f"!! ERROR: Could not download '{dataset_id}'. Please check the name "
                f"and your connection. Error: {e}"
            )
            return False


def setup_datasets():
    print("--- Starting Minari Dataset Setup ---")
    try:
        local_datasets = set(minari.list_local_datasets().keys())
    except Exception as e:
        print(f"!! ERROR: Could not list local Minari datasets. Is Minari installed correctly? Error: {e}")
        return

    for i, config in enumerate(DATASET_CONFIG):
        target_id = config["target_id"]
        task_type = config["type"]
        print(f"\n[{i+1}/{len(DATASET_CONFIG)}] Processing '{target_id}'...")

        if target_id in local_datasets:
            print(f"==> Target dataset '{target_id}' already exists. Skipping.")
            continue

        if task_type == "download":
            download_if_missing(target_id, local_datasets)

        elif task_type == "combine":
            source_ids = config["source_ids"]
            datasets_to_combine = []
            all_sources_ready = True

            # First, ensure all source datasets are available
            for sid in source_ids:
                if not download_if_missing(sid, local_datasets):
                    all_sources_ready = False
                    break # Stop if a required source can't be downloaded

            # If all sources are ready, proceed with combination
            if all_sources_ready:
                print(f"--> Loading source datasets to create '{target_id}'...")
                try:
                    for sid in source_ids:
                        datasets_to_combine.append(minari.load_dataset(sid))
                    
                    print(f"--> Combining datasets...")
                    minari.combine_datasets(datasets_to_combine, new_dataset_id=target_id)
                    print(f"==> Successfully created '{target_id}'!")
                except Exception as e:
                    print(f"!! ERROR: Could not create '{target_id}'. Error: {e}")
            else:
                print(f"!! Skipping combination for '{target_id}' due to missing source datasets.")

    print("\n--- Dataset setup complete. ---")


if __name__ == "__main__":
    setup_datasets()
