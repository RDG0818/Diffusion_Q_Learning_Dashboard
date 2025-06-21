import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.data_loader import load_new_experiments, load_dagger_experiments
from pathlib import Path
import numpy as np

NAME = "Comparative Analysis"
ENVIRONMENTS = ["hopper", "walker2d", "swimmer", "halfcheetah", "ant"]
CSV_METRICS = [
    'avg_reward', 'std_reward', 'avg_ep_length',
    'actor_loss', 'critic_loss', 'bc_loss', 'ql_loss',
    'avg_q_batch', 'avg_q_policy',
    'actor_grad_norm', 'critic_grad_norm'
]
LABELS = {m: m.replace('_', ' ').title() for m in CSV_METRICS}

# Aggregate and (optionally) smooth data

def aggregate(df, metric, smooth=True):
    df = df.sort_values('step')
    if smooth:
        df[metric] = df.groupby('experiment')[metric].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    # base key without seed
    df['base'] = df['experiment'].apply(lambda x: '-'.join(x.split('-')[:-1]))
    agg = df.groupby(['base', 'step'])[metric].agg(['mean', 'min', 'max']).reset_index()
    agg.columns = ['base', 'step', 'mean', 'min', 'max']
    return agg

# Plot with shaded bounds

def plot_with_bounds(df, label):
    fig = go.Figure()
    for base in df['base'].unique():
        sub = df[df['base'] == base]
        fig.add_trace(go.Scatter(x=sub['step'], y=sub['mean'], mode='lines', name=f"{base}"))
        fig.add_trace(go.Scatter(
            x=sub['step'], y=sub['max'],
            mode='lines', line=dict(width=0), hoverinfo='none', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=sub['step'], y=sub['min'],
            mode='lines', fill='tonexty', line=dict(width=0), hoverinfo='none', showlegend=False
        ))
    fig.update_layout(
        xaxis_title='Step', yaxis_title=label,
        template='plotly_dark'
    )
    return fig

# Main app
def app(df=None):
    st.markdown(f"<h1 style='text-align:center;'>{NAME}</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align:center; font-size:16px;'>
        Visualize averaged performance metrics (with min/max bounds) across multiple seeds for a chosen environment.
        """,
        unsafe_allow_html=True
    )
    st.divider()

    # Sidebar: config
    st.sidebar.header("Configuration")
    base_dir = st.sidebar.text_input("Results Directory", "results")
    env = st.sidebar.selectbox("Environment", ENVIRONMENTS)

    # Discover all runs for env
    try:
        runs = sorted([d.name for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith(f"{env}-")])
    except Exception:
        st.sidebar.error(f"Cannot access '{base_dir}'")
        return
    if not runs:
        st.error(f"No runs found for environment '{env}' in '{base_dir}'")
        return

    # Load and filter data
    data_new = load_new_experiments(base_dir)
    data_dagger = load_dagger_experiments(base_dir)
    dfs = []
    if data_new is not None:
        df = data_new.rename(columns={'experiment_name':'experiment'})
        df = df[df['experiment'].isin(runs)]
        dfs.append(df)
    if data_dagger is not None:
        df2 = data_dagger.rename(columns={'experiment_name':'experiment'})
        df2 = df2[df2['experiment'].isin(runs)]
        dfs.append(df2)
    if not dfs:
        st.error("No data loaded.")
        return
    data = pd.concat(dfs, ignore_index=True)

    # Layout: two columns, then dagger, then extras
    col1, col2 = st.columns(2)

    # Left: avg_reward, actor_loss
    with col1:
        for metric in ['avg_reward', 'actor_loss']:
            agg = aggregate(data.copy(), metric)
            fig = plot_with_bounds(agg, LABELS[metric])
            st.plotly_chart(fig, use_container_width=True)

    # Right: avg_ep_length, critic_loss
    with col2:
        for metric in ['avg_ep_length', 'critic_loss']:
            agg = aggregate(data.copy(), metric)
            fig = plot_with_bounds(agg, LABELS[metric])
            st.plotly_chart(fig, use_container_width=True)

    # DAgger comparison
    st.markdown("### DAgger Model Comparison")
    dagger_files = list(Path(base_dir).glob("**/*student_policy_dagger_eval*.csvh"))
    if dagger_files:
        file = st.selectbox("Select DAgger CSV", dagger_files)
        df_d = pd.read_csv(file)
        df_d['model'] = df_d['model'].replace({'student':'Distilled MLP','teacher':'Diffusion QL'})
        col = st.selectbox("Compare by", ['reward','length','time'])
        teacher_avg = df_d[df_d['model']=='Diffusion QL'][col].mean()
        fig = px.box(
            df_d[df_d['model']=='Distilled MLP'], x='model', y=col, color='model', points='all', template='plotly_dark'
        )
        fig.add_hline(y=teacher_avg, line_dash='dot', line_color='red', line_width=3,
                      annotation_text='Diffusion QL Avg', annotation_position='top left')
        st.plotly_chart(fig, use_container_width=True)

    # Additional metrics
    st.markdown("### Additional Metric Comparison")
    extra = st.multiselect("Select Extra Metrics", [m for m in CSV_METRICS if m not in ['avg_reward','actor_loss','avg_ep_length','critic_loss']])
    smooth = st.checkbox("Apply Smoothing", value=True)
    for metric in extra:
        agg = aggregate(data.copy(), metric, smooth=smooth)
        fig = plot_with_bounds(agg, LABELS[metric])
        st.plotly_chart(fig, use_container_width=True)
