# modules/pages/home.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# NAME attribute is required by your app.py to build the navigation.
NAME = "Home"

def create_returns_graph(df: pd.DataFrame, environment_name: str):
    """
    Creates a Plotly graph of returns for a SPECIFIC environment.
    """
    df_env = df[df['environment'] == environment_name]
    
    if df_env.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data found for {environment_name}", template="plotly_dark")
        return fig
        
    stats_df = df_env.groupby('step')['norm_return'].agg(['mean', 'min', 'max']).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stats_df['step'], y=stats_df['max'], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=stats_df['step'], y=stats_df['min'], mode='lines', line=dict(width=0),
        fillcolor='rgba(255, 153, 0, 0.2)', fill='tonexty', showlegend=False, name='Min/Max Range'))
    fig.add_trace(go.Scatter(x=stats_df['step'], y=stats_df['mean'], mode='lines', name='Mean Return',
        line=dict(color='#FF9900', width=3),
        hovertemplate='<b>Step</b>: %{x}<br><b>Mean Norm. Return</b>: %{y:.2f}<extra></extra>'))
    
    fig.update_layout(xaxis_title="Training Steps", yaxis_title="Normalized Return", template="plotly_dark",
        legend=dict(x=0.01, y=0.99), margin=dict(l=40, r=40, t=30, b=40))
    return fig

def create_loss_graph(df: pd.DataFrame, environment_name: str):
    """
    Creates a dual-axis Plotly graph of losses for a SPECIFIC environment.
    """
    df_env = df[df['environment'] == environment_name]

    if df_env.empty:
        return go.Figure()

    stats_df = df_env.groupby('step')[['actor_loss', 'critic_loss']].mean().reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=stats_df['step'], y=stats_df['actor_loss'], name='Actor Loss',
        line=dict(color='#1f77b4', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=stats_df['step'], y=stats_df['critic_loss'], name='Critic Loss',
        line=dict(color='#d62728', width=2)), secondary_y=True)

    # --- ROBUST TICK SYNCHRONIZATION FIX ---
    # Manually calculate tick values to force synchronization
    num_ticks = 6 # Define the desired number of ticks
    
    # Calculate ticks for the primary y-axis (Actor Loss)
    actor_min, actor_max = stats_df['actor_loss'].min(), stats_df['actor_loss'].max()
    actor_tick_values = np.linspace(actor_min, actor_max, num_ticks)

    # Calculate ticks for the secondary y-axis (Critic Loss)
    critic_min, critic_max = stats_df['critic_loss'].min(), stats_df['critic_loss'].max()
    critic_tick_values = np.linspace(critic_min, critic_max, num_ticks)

    fig.update_yaxes(title_text="Actor Loss", secondary_y=False, 
                     tickmode='array', tickvals=actor_tick_values, tickformat=".2f")
    fig.update_yaxes(title_text="Critic Loss", secondary_y=True,
                     tickmode='array', tickvals=critic_tick_values, tickformat=".1f")
    # --- END OF FIX ---
    
    fig.update_layout(xaxis_title="Training Steps", template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def app(df):
    # This part is correct and remains the same
    st.sidebar.header("Project Details")
    st.sidebar.info(
        """
        This project enhances the original Diffusion Q-Learning codebase. My key contributions include:

        **Modernization:** Updating libraries for improved stability and compatibility.

        **Performance:** Re-engineering the pipeline for faster, distributed training.
        
        **Analysis:** Developing this dashboard for interactive visual insights.
        """
    )
    st.sidebar.warning(
        "**Attribution:** I am not affiliated with the original research team. "
        "This work is a re-implementation created for analysis and demonstration. "
        "Full credit for the foundational algorithm belongs to the original authors."
    )
    st.sidebar.markdown("""
        **Quick Links:**
        - [GitHub](https://github.com/RDG0818/Offline-RL-Trajectory-Diffusion-Policy)
        - [Original Research Paper](https://arxiv.org/pdf/2208.06193)
        - [Author's LinkedIn](https://www.linkedin.com/in/ryan-goodwin818/)
    """)
    st.markdown(f"<h1 style='text-align: center;'>Diffusion Q-Learning Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
    A visualization and analysis suite for exploring offline RL experiments with Diffusion Q-Learning, meant to examine training dynamics, sample efficiency, and model behavior.
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>Core Concepts & Objectives</h3>", unsafe_allow_html=True)
            
            st.markdown("- **Policy as a Diffusion Model:** The policy is represented as a conditional diffusion model that generates actions by reversing a noising process.  It starts with random noise and iteratively refines it over N steps to produce a final action, conditioned on the current state.  This approach allows for a highly expressive policy that can capture complex, multi-modal distributions often found in offline datasets. ")
            st.latex(r'''
            \pi_{\theta}(a|s) = p_{\theta}(a^{0:N}|s) = \mathcal{N}(a^{N};0,I)\prod_{i=1}^{N}p_{\theta}(a^{i-1}|a^{i},s)
            ''')
            st.markdown("---")
            
            st.markdown("- **Implicit Behavior Cloning:** The diffusion model's training objective naturally serves as a behavior-cloning (BC) loss.  It is a powerful distribution matching technique that encourages the model to generate actions from the same distribution as the data it was trained on.  This acts as an implicit policy regularizer, constraining the policy to the data distribution. ")
            st.latex(r'''
            \mathcal{L}_{\text{d}}(\theta)=\mathbb{E}_{i,(s,a),\epsilon}[||\epsilon-\epsilon_{\theta}(\sqrt{\overline{\alpha}_{i}}a+\sqrt{1-\overline{\alpha}_{i}}\epsilon,s,i)||^{2}]
            ''')
            st.markdown("---")
            
            st.markdown("- **Q-Function Guided Policy Improvement:** To learn a policy better than the one that generated the data, Q-learning is injected directly into the training process.  A policy improvement term, which maximizes the expected Q-value of generated actions, is added to the loss function.  This guides the diffusion policy to seek optimal, high-value actions within the regularized region defined by the BC loss. ")
            st.latex(r'''
            \mathcal{L}_{\text{q}}(\theta) = - \eta \cdot \mathbb{E}_{s \sim \mathcal{D}, a^0 \sim \pi_\theta}[Q_\phi(s, a^0)]
            ''')

        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>Training Loss Analysis</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='text-align: center; font-size: 0.9rem; color: #fafafaB3;'>
            The <b>Actor Loss</b> typically shows an exponential decay as the policy learns. The <b>Critic Loss</b> often humps as it first learns to differentiate values and then converges.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

            # --- ROBUST CENTERING FIX ---
            # Use columns to center the radio button widget
            _ , radio_col, _ = st.columns([1, 2, 1])
            with radio_col:
                env_choice = st.radio("Select Environment:",
                    ["halfcheetah-medium-expert-v2", "walker2d-medium-expert-v2", "hopper-medium-expert-v2"], 
                    horizontal=True, label_visibility="collapsed")
            # --- END OF FIX ---
            
            st.markdown(f"<h5 style='text-align: center;'>Average Training Losses: {env_choice}</h5>", unsafe_allow_html=True)
            
            loss_fig = create_loss_graph(df, env_choice)
            st.plotly_chart(loss_fig, use_container_width=True)

        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>Dataset Overview & Resampling</h3>", unsafe_allow_html=True)
            total_transitions_list = [1_998_000, 1_998_318, 1_998_966]
            avg_dataset_size = np.mean(total_transitions_list)
            transitions_sampled_per_env = 102_400_000
            
            col1, col2, col3 = st.columns([3, 3, 3], gap="large")
            with col1:
                st.metric("Avg. Total Transitions", f"~{avg_dataset_size / 1e6:.1f}M")
                st.caption("Unique data points")
            with col2:
                st.metric("Samples Drawn", f"{transitions_sampled_per_env / 1e6:.1f}M")
                st.caption("Per environment")
            with col3:
                resampling_factor = transitions_sampled_per_env / avg_dataset_size
                st.metric("Avg. Resampling Factor", f"~{resampling_factor:.1f}x")
                st.caption("Each point seen")

    with right_col:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>Performance Visualizations</h3>", unsafe_allow_html=True)
            
            st.markdown("<h5 style='text-align: center;'>halfcheetah-medium-expert-v2</h5>", unsafe_allow_html=True)
            returns_fig_hc = create_returns_graph(df, "halfcheetah-medium-expert-v2")
            st.plotly_chart(returns_fig_hc, use_container_width=True)
            
            st.divider()

            st.markdown("<h5 style='text-align: center;'>walker2d-medium-expert-v2</h5>", unsafe_allow_html=True)
            returns_fig_w2d = create_returns_graph(df, "walker2d-medium-expert-v2")
            st.plotly_chart(returns_fig_w2d, use_container_width=True)

            st.divider()
            
            st.markdown("<h5 style='text-align: center;'>hopper-medium-expert-v2</h5>", unsafe_allow_html=True)
            returns_fig_hp = create_returns_graph(df, "hopper-medium-expert-v2")
            st.plotly_chart(returns_fig_hp, use_container_width=True)