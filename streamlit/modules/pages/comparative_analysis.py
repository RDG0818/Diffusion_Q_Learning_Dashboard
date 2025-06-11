import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.data_loader import load_new_experiments, load_dagger_experiments
from pathlib import Path
import numpy as np
import re

NAME = "Comparative Analysis"
ENVIRONMENTS = ["hopper", "walker2d", "swimmer", "halfcheetah", "ant"]
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

def smooth_and_aggregate(df, metric):
    df = df.sort_values('step')
    df[metric] = df.groupby('experiment')[metric].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['base'] = df['experiment'].apply(lambda x: '-'.join(x.split('-')[:-1]))
    agg = df.groupby(['base', 'step'])[metric].agg(['mean', 'min', 'max']).reset_index()
    return agg

def plot_with_bounds(df, metric, label):
    fig = go.Figure()
    bases = df['base'].unique()
    for b in bases:
        sub = df[df['base'] == b]
        fig.add_trace(go.Scatter(x=sub['step'], y=sub['mean'], mode='lines', name=b))
        fig.add_trace(go.Scatter(x=sub['step'], y=sub['max'], mode='lines', name=f'{b} max', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=sub['step'], y=sub['min'], mode='lines', name=f'{b} min', fill='tonexty', line=dict(width=0), showlegend=False))
    fig.update_layout(title=label, xaxis_title='Step', yaxis_title=label, template='plotly_dark')
    return fig

def app(df=None):
    st.markdown(f"<h1 style='text-align:center;'>{NAME}</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align:center; font-size:16px;'>
        Compare baseline and DAgger-augmented runs on the same environment. Select an environment,
        pick experiment folders, and visualize reward, losses, and other metrics.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    # Sidebar selection
    st.sidebar.header("Configuration")
    base_dir = st.sidebar.text_input("Results Directory", "results")
    env = st.sidebar.selectbox("Environment", ENVIRONMENTS)

    try:
        runs = [d.name for d in Path(base_dir).iterdir() if d.is_dir() and env.lower() in d.name.lower()]
    except Exception:
        st.sidebar.error(f"Cannot access '{base_dir}'")
        return

    selected = st.sidebar.multiselect("Select Runs", runs, default=runs[:2])
    if not selected:
        st.warning("Please select at least one run.")
        return

    # Load data
    data_new = load_new_experiments(base_dir)
    data_dagger = load_dagger_experiments(base_dir)

    data_frames = []
    if data_new is not None and not data_new.empty:
        data_new = data_new.rename(columns={'experiment_name': 'experiment'})
        data_new['type'] = 'baseline'
        data_new = data_new[data_new['experiment'].isin(selected)]
        data_frames.append(data_new)
    if data_dagger is not None and not data_dagger.empty:
        data_dagger = data_dagger.rename(columns={'experiment_name': 'experiment'})
        data_dagger['type'] = 'dagger'
        data_dagger = data_dagger[data_dagger['experiment'].isin(selected)]
        data_frames.append(data_dagger)

    if not data_frames:
        st.error("No matching data found.")
        return

    data = pd.concat(data_frames, ignore_index=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        for metric in ['avg_reward', 'actor_loss']:
            st.markdown(f"### {LABELS.get(metric, metric)}")
            agg = smooth_and_aggregate(data.copy(), metric)
            fig = plot_with_bounds(agg, metric, LABELS.get(metric, metric))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        for metric in ['avg_ep_length', 'critic_loss']:
            st.markdown(f"### {LABELS.get(metric, metric)}")
            agg = smooth_and_aggregate(data.copy(), metric)
            fig = plot_with_bounds(agg, metric, LABELS.get(metric, metric))
            st.plotly_chart(fig, use_container_width=True)

    # DAgger Comparison
    st.markdown("### DAgger Model Comparison")
    dagger_files = list(Path(base_dir).glob("**/*student_policy_dagger_eval*.csvh"))
    if dagger_files:
        file = st.selectbox("Select DAgger CSV", dagger_files)
        df_dagger = pd.read_csv(file)

        df_dagger['model'] = df_dagger['model'].replace({'student': 'Distilled MLP', 'teacher': 'Diffusion QL'})
        col = st.selectbox("Compare by", ['reward', 'length', 'time'])
        teacher_avg = df_dagger[df_dagger['model'] == 'Diffusion QL'][col].mean()

        fig = px.box(
            df_dagger[df_dagger['model'] == 'Distilled MLP'],
            x='model', y=col, color='model', points='all', template='plotly_dark',
            title=f"Distilled MLP vs Diffusion QL: {col.capitalize()}"
        )
        fig.add_hline(y=teacher_avg, line_dash='dot', line_color='red', line_width=3,
                      annotation_text='Diffusion QL Avg', annotation_position='top left')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No DAgger evaluation CSVs found.")

    # Extra metrics
    st.markdown("### Additional Metric Comparison")
    extra_metrics = st.multiselect(
        "Select Extra Metrics",
        [m for m in CSV_METRICS if m not in ['avg_reward', 'actor_loss', 'avg_ep_length', 'critic_loss']]
    )
    smooth = st.checkbox("Apply Smoothing", value=True)

    for metric in extra_metrics:
        agg = smooth_and_aggregate(data.copy(), metric) if smooth else None
        fig = plot_with_bounds(agg, metric, LABELS.get(metric, metric))
        st.plotly_chart(fig, use_container_width=True)
