# modules/data_loader.py
import pandas as pd
import streamlit as st
import csv
from pathlib import Path
import re

def _parse_single_csv(filepath: Path) -> pd.DataFrame | None:
    # This helper function remains the same
    processed_data = []
    try:
        with open(filepath, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            while True:
                try:
                    train_row, eval_row = next(reader), next(reader)
                    if not train_row or not eval_row: continue
                    epoch = int(train_row[0])
                    processed_data.append({
                        'epoch': epoch, 'step': epoch * 1000,
                        'actor_loss': float(train_row[1]), 'critic_loss': float(train_row[2]),
                        'return': float(eval_row[0]), 'norm_return': float(eval_row[1])
                    })
                except StopIteration: break
                except (ValueError, IndexError): continue
    except Exception: return None
    return pd.DataFrame(processed_data)

@st.cache_data
def load_all_experiments(base_path_str: str) -> pd.DataFrame | None:
    """
    Finds all 'progress.csv' files for different environments and seeds,
    loads them, and combines them into a single master DataFrame.
    """
    base_path = Path(base_path_str)
    if not base_path.is_dir():
        st.error(f"Error: Base directory not found at '{base_path_str}'")
        return None

    all_dfs = []
    # Use glob to find all progress.csv files recursively
    csv_files = list(base_path.glob("**/progress.csv"))
    
    if not csv_files:
        st.warning(f"No 'progress.csv' files found in subdirectories of '{base_path_str}'.")
        return pd.DataFrame()

    for filepath in csv_files:
        # Extract environment name (e.g., 'halfcheetah-medium-expert-v2')
        try:
            # Assumes path is like 'results/halfcheetah-medium-expert-v2-seed0/progress.csv'
            env_name = filepath.parent.name.rsplit('-', 1)[0]
        except Exception:
            env_name = "unknown"

        # Extract seed number
        match = re.search(r'-seed(\d+)', str(filepath))
        seed = int(match.group(1)) if match else -1

        df = _parse_single_csv(filepath)
        if df is not None and not df.empty:
            df['environment'] = env_name
            df['seed'] = seed
            all_dfs.append(df)

    if not all_dfs:
        st.error("Could not load any valid data from the found CSV files.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)