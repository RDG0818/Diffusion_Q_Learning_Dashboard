import minari

BASE_ENVIRONMENTS = ["halfcheetah", "hopper", "walker2d", "ant", "swimmer"]
SOURCE_QUALITIES = ["medium", "expert"]
NAMESPACE = "mujoco"


def setup_datasets():
    """
    A script to download source datasets and combine them to create
    the 'medium-expert' variants.
    """

    try:
        local_datasets = set(minari.list_local_datasets().keys())
    except Exception as e:
        print(
            f"Could not list local Minari datasets. Check your Minari installation. Error: {e}"
        )
        local_datasets = set()

    for env in BASE_ENVIRONMENTS:
        for quality in SOURCE_QUALITIES:
            dataset_id = f"{NAMESPACE}/{env}/{quality}-v0"

            if dataset_id in local_datasets:
                print(f"'{dataset_id}' already exists locally. Skipping download.")
            else:
                try:
                    print(f"Downloading '{dataset_id}'...")
                    minari.download_dataset(dataset_id=dataset_id)
                    print(f"Successfully downloaded '{dataset_id}'.")
                except Exception as e:
                    print(
                        f"ERROR: Could not download '{dataset_id}'. Please check the name and your connection. Error: {e}"
                    )

    try:
        local_datasets = set(minari.list_local_datasets().keys())
    except Exception:
        local_datasets = set()

    for env in BASE_ENVIRONMENTS:
        combined_id = f"{NAMESPACE}/{env}/medium-expert-v0"
        source_ids = [f"{NAMESPACE}/{env}/medium-v0", f"{NAMESPACE}/{env}/expert-v0"]

        if combined_id in local_datasets:
            print(
                f"Combined dataset '{combined_id}' already exists. Skipping combination."
            )
            continue

        datasets_to_combine = []
        all_sources_found = True
        print(f"Loading source datasets for '{combined_id}'...")
        for sid in source_ids:
            if sid in local_datasets:
                try:
                    loaded_dataset = minari.load_dataset(sid)
                    datasets_to_combine.append(loaded_dataset)
                except Exception as e:
                    print(f"ERROR: Failed to load '{sid}'. Error: {e}")
                    all_sources_found = False
                    break
            else:
                print(
                    f"WARNING: Source dataset '{sid}' not found locally. Cannot create combined dataset."
                )
                all_sources_found = False
                break

        if all_sources_found:
            print(f"Combining datasets to create '{combined_id}'...")
            try:
                minari.combine_datasets(datasets_to_combine, new_dataset_id=combined_id)
                print(f"Successfully created '{combined_id}'!")
            except Exception as e:
                print(f"ERROR: Could not create '{combined_id}'. Error: {e}")

    print("\nDataset setup complete.")


if __name__ == "__main__":
    setup_datasets()
