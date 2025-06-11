import pandas as pd
import streamlit as st
import csv
from pathlib import Path
import re


def _parse_old_csv(filepath: Path) -> pd.DataFrame | None:
    """
    Parse a CSV file in the old alternating train/eval row format.
    """
    processed_data = []
    try:
        with open(filepath, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            # Skip header-like row if present
            first = next(reader)
            if not re.match(r"\d+", first[0]):
                # assume header, continue with data rows
                pass
            else:
                infile.seek(0)
                reader = csv.reader(infile)
            while True:
                try:
                    train_row = next(reader)
                    eval_row = next(reader)
                except StopIteration:
                    break
                except Exception:
                    continue
                if not train_row or not eval_row:
                    continue
                try:
                    epoch = int(train_row[0])
                    processed_data.append({
                        'epoch': epoch,
                        'step': epoch * 1000,
                        'actor_loss': float(train_row[1]),
                        'critic_loss': float(train_row[2]),
                        'return': float(eval_row[0]),
                        'norm_return': float(eval_row[1])
                    })
                except Exception:
                    continue
    except Exception:
        return None

    if not processed_data:
        return None
    return pd.DataFrame(processed_data)


def _load_new_csv(filepath: Path) -> pd.DataFrame | None:
    """
    Load a new-style CSV with header 'step,avg_reward,...'.
    """
    try:
        df = pd.read_csv(filepath)
        required = {'step', 'avg_reward', 'std_reward', 'avg_ep_length',
                    'actor_loss', 'critic_loss', 'bc_loss', 'ql_loss',
                    'avg_q_batch', 'avg_q_policy',
                    'actor_grad_norm', 'critic_grad_norm'}
        if not required.issubset(df.columns):
            print(f"Skipping file {filepath.name}: Missing columns.")
            print(f"  Missing: {required - set(df.columns)}")
            print(f"  Found:   {list(df.columns)}")
            return None
        return df
    except Exception as e:
        print(f"Could not read file {filepath.name} due to an error: {e}")
        return None

@st.cache_data
def load_old_experiments(base_path_str: str) -> pd.DataFrame:
    """
    Load experiments stored in the old CSV format from a base directory.
    """
    base_path = Path(base_path_str)
    if not base_path.is_dir():
        st.error(f"Old results dir not found: '{base_path_str}'")
        return pd.DataFrame()
    all_dfs = []
    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        for filepath in exp_dir.glob("**/progress.csv"):
            df = _parse_old_csv(filepath)
            if df is not None and not df.empty:
                df['experiment_name'] = name
                all_dfs.append(df)
    if not all_dfs:
        st.warning(f"No old-format CSVs in '{base_path_str}'.")
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

@st.cache_data
def load_new_experiments(base_path_str: str) -> pd.DataFrame:
    """
    Load experiments stored in the new header-based CSV format from a base directory.
    """
    base_path = Path(base_path_str)
    if not base_path.is_dir():
        st.error(f"New results dir not found: '{base_path_str}'")
        return pd.DataFrame()
    all_dfs = []
    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        for filepath in exp_dir.glob("**/*.csv"):
            df = _load_new_csv(filepath)
            if df is not None and not df.empty:
                df['experiment_name'] = name
                all_dfs.append(df)
    if not all_dfs:
        st.warning(f"No new-format CSVs in '{base_path_str}'.")
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

@st.cache_data
def load_dagger_experiments(base_path_str: str) -> pd.DataFrame:
    """
    Load DAgger evaluation data from student experiments in CSV format.
    Expects files named '*_dagger_eval.csvh' under each experiment directory.
    """
    base_path = Path(base_path_str)
    if not base_path.is_dir():
        st.error(f"DAgger results dir not found: '{base_path_str}'")
        return pd.DataFrame()
    all_dfs = []
    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        # find dagger evaluation CSVs
        for filepath in exp_dir.glob("**/*_dagger_eval.csvh"):
            try:
                df = pd.read_csv(filepath)
                # ensure expected columns
                required = {'model', 'episode', 'reward', 'length', 'time'}
                if not required.issubset(df.columns):
                    st.warning(f"Skipping {filepath.name}: missing columns {required - set(df.columns)}")
                    continue
                df['experiment_name'] = name
                all_dfs.append(df)
            except Exception as e:
                st.warning(f"Failed to load {filepath.name}: {e}")
                continue
    if not all_dfs:
        st.warning(f"No DAgger evaluation CSVs in '{base_path_str}'.")
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)