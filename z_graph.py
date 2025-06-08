import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from typing import List, Dict, Tuple

def load_and_process_npy(filepath: str) -> pd.DataFrame:
    """
    Loads an evaluation .npy file and converts it into a pandas DataFrame.
    It also creates a 'Timesteps' column by multiplying 'Epoch' by 1000.

    Args:
        filepath (str): The path to the 'eval.npy' file.

    Returns:
        pd.DataFrame: A DataFrame with named columns for easy access.
    """
    try:
        raw_data = np.load(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return pd.DataFrame()

    columns = [
        'Evaluation', 'Evaluation Std', 'Norm Evaluation', 'Norm Evaluation Std',
        'BC Loss', 'QL Loss', 'Actor Loss', 'Critic Loss', 'Epoch'
    ]
    df = pd.DataFrame(data=raw_data, columns=columns)
    # Create the 'Timesteps' column for the new x-axis
    df['Timesteps'] = df['Epoch'] * 1000
    return df

def plot_evaluation_data(all_data: List[Tuple[pd.DataFrame, str]],
                         y_column: str,
                         save_path: str,
                         title: str,
                         rolling_window: int,
                         colors: List[str]) -> None:
    """
    Generates and saves a plot from multiple evaluation datasets with custom styling.

    Args:
        all_data (list): A list of tuples, where each tuple is (DataFrame, legend_label).
        y_column (str): The name of the column to plot on the y-axis.
        save_path (str): The path where the plot image will be saved.
        title (str): The title for the plot.
        rolling_window (int): The window size for the rolling average.
        colors (list): A list of color strings to use for plotting.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, (df, label) in enumerate(all_data):
        if y_column not in df.columns:
            print(f"Warning: Column '{y_column}' not found in data for label '{label}'. Skipping.")
            continue
        
        # Assign a color from the provided list, cycling if necessary
        line_color = colors[i % len(colors)]

        x_axis = df['Timesteps']
        y_axis = df[y_column]
        std_column_name = f"{y_column} Std"

        # Apply rolling average if window size is greater than 1
        if rolling_window > 1:
            y_axis = y_axis.rolling(window=rolling_window, min_periods=1).mean()
            if std_column_name in df.columns:
                std_dev = df[std_column_name].rolling(window=rolling_window, min_periods=1).mean()
        else:
            if std_column_name in df.columns:
                std_dev = df[std_column_name]

        # Plot the main metric line
        ax.plot(x_axis, y_axis, color=line_color, linestyle='-', label=label, linewidth=3.0)

        # Plot the standard deviation if it exists
        if std_column_name in df.columns and std_dev is not None:
            ax.fill_between(x_axis, y_axis - std_dev, y_axis + std_dev, alpha=0.2, color=line_color)
            ax.plot(x_axis, y_axis - std_dev, color=line_color, linestyle='-', alpha=0.4, linewidth=1.0)
            ax.plot(x_axis, y_axis + std_dev, color=line_color, linestyle='-', alpha=0.4, linewidth=1.0)

    # Styling
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Timesteps', fontsize=14)
    ax.set_ylabel(y_column, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, frameon=False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    try:
        plt.savefig(save_path)
        print(f"Graph successfully saved to: {save_path}")
    except Exception as e:
        print(f"Error saving graph: {e}")
    finally:
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot evaluation results from one or more 'eval.npy' files.")
    y_axis_choices = ['Evaluation', 'Norm Evaluation', 'BC Loss', 'QL Loss', 'Actor Loss', 'Critic Loss']
    
    parser.add_argument('y_column', type=str, choices=y_axis_choices, help="The primary metric to plot.")
    parser.add_argument('-f', '--files', nargs='+', required=True, help="Paths to the .npy files to plot.")
    parser.add_argument('-l', '--labels', nargs='+', required=True, help="Legend labels for each file, in the same order.")
    parser.add_argument('-c', '--colors', nargs='+', help="Colors for each file, in the same order (e.g., '#001f3f' 'red').")
    parser.add_argument('-w', '--window', type=int, default=1, help="Rolling average window size. Default is 1 (no smoothing).")
    parser.add_argument('-t', '--title', type=str, default=None, help="Custom title for the plot.")
    parser.add_argument('-o', '--output', type=str, default=None, help="Output filename for the saved plot.")
    
    args = parser.parse_args()

    if len(args.files) != len(args.labels):
        raise ValueError("The number of --files must match the number of --labels.")
    if args.colors and len(args.files) != len(args.colors):
        raise ValueError("If providing --colors, the number must match the number of --files.")
        
    SAVE_FOLDER = 'graphs'
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    
    plot_data = []
    for f, l in zip(args.files, args.labels):
        df = load_and_process_npy(f)
        if not df.empty:
            plot_data.append((df, l))

    if not plot_data:
        print("No valid data found to plot.")
    else:
        # Define a default palette if no colors are provided
        colors = args.colors if args.colors else [
            '#001f3f', '#2ECC40', '#FF4136', '#B10DC9', 
            '#FF851B', '#0074D9', '#3D9970'
        ]
        
        plot_title = args.title if args.title else f'Evaluation: {args.y_column} vs. Timesteps'
        output_name = args.output if args.output else f"{args.y_column.replace(' ', '_')}_comparison.png"
        full_save_path = os.path.join(SAVE_FOLDER, output_name)
        
        plot_evaluation_data(plot_data, args.y_column, full_save_path, plot_title, args.window, colors)
