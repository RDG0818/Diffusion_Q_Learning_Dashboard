import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.data_loader import load_new_experiments
from pathlib import Path
import numpy as np

NAME = "Comparative Analysis"
CSV_METRICS = [
    'avg_reward', 'std_reward', 'avg_ep_length',
    'actor_loss', 'critic_loss', 'bc_loss', 'ql_loss',
    'avg_q_batch', 'avg_q_policy',
    'actor_grad_norm', 'critic_grad_norm'
]
LABELS = {
    'avg_reward': 'Avg Reward',
    'std_reward': 'Reward Std',
    'avg_ep_length': 'Episode Length',
    'actor_loss': 'Actor Loss',
    'critic_loss': 'Critic Loss',
    'bc_loss': 'BC Loss',
    'ql_loss': 'QL Loss',
    'avg_q_batch': 'Avg Q (Batch)',
    'avg_q_policy': 'Avg Q (Policy)',
    'actor_grad_norm': 'Actor Grad Norm',
    'critic_grad_norm': 'Critic Grad Norm'
}


def app(df=None):
    st.markdown(f"<h1 style='text-align:center;'>{NAME}</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align:center; font-size:16px;'>
        Select experiment runs and compare their performance and learning behavior. 
        Display line charts for all key metrics over training steps.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    # Sidebar: directory and run selection
    st.sidebar.header("Data & Runs Selection")
    base_dir = st.sidebar.text_input("Results Directory", "results")
    runs = []
    try:
        runs = [d.name for d in Path(base_dir).iterdir() if d.is_dir()]
    except Exception:
        st.sidebar.error(f"Cannot access '{base_dir}'")
    selected = st.sidebar.multiselect("Select Runs", runs, default=runs[:2])
    if not selected:
        st.warning("Please select at least one run.")
        return

    # Load and filter data
    data_all = load_new_experiments(base_dir)
    if data_all is None or data_all.empty:
        st.error("No data found in specified directory.")
        return
    data = data_all[data_all['experiment_name'].isin(selected)].copy()
    data.rename(columns={'experiment_name':'experiment'}, inplace=True)
    if data.empty:
        st.error("Selected runs contain no valid data.")
        return

    # Plot line chart for each metric
    for metric in CSV_METRICS:
        st.markdown(f"<h3 style='text-align:center;'>{LABELS.get(metric, metric)} Over Training</h3>", unsafe_allow_html=True)
        fig = px.line(
            data, x='step', y=metric, color='experiment',
            labels={'step':'Step', metric: LABELS.get(metric, metric)},
            title=LABELS.get(metric, metric), template='plotly_dark'
        )
        fig.update_traces(line=dict(width=2))
        st.plotly_chart(fig, use_container_width=True)
    return

