# modules/pages/denoising_process.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# NAME attribute is required by app.py to build the navigation.
NAME = "Denoising Process"

# --- Placeholder Function for Denoising Logic ---
# You will replace this with your actual model's generative process.
def generate_denoising_trajectory(start_state, num_timesteps=50, num_points=100, action_dim=2):
    """
    Generates a trajectory of points moving from noise to a target action.
    
    This is a PLACEHOLDER. You should replace the logic inside the loop
    with calls to your actual trained diffusion model (epsilon_theta).

    Args:
        start_state: The state for which to generate an action (currently unused in this placeholder).
        num_timesteps: The number of denoising steps (T).
        num_points: The number of action samples to visualize.
        action_dim: The dimensionality of the action space (must be 2 for this scatter plot).

    Returns:
        A Pandas DataFrame with columns ['x', 'y', 'timestep', 'point_id'].
    """
    # 1. Define a placeholder target action. In your real implementation,
    # the model implicitly learns this from the Q-function and dataset.
    target_action = np.array([0.7, -0.4]) 
    
    # 2. Start with a cloud of pure Gaussian noise. This is our state at t=T.
    current_points = np.random.randn(num_points, action_dim)
    
    # 3. Store the full trajectory for plotting.
    trajectory_data = []

    # 4. Loop backwards from T down to 0, simulating the denoising process.
    for t in reversed(range(num_timesteps + 1)):
        # --- !!! REPLACE THIS BLOCK WITH YOUR MODEL !!! ---
        # In a real model, you would do:
        # noise_pred = model.epsilon_theta(current_points, t, start_state)
        # current_points = model.denoise_step(current_points, noise_pred, t)
        
        # Placeholder logic: a simple linear interpolation towards the target.
        # This simulates the points getting "cleaner" over time.
        alpha = t / num_timesteps
        noise_component = alpha * np.random.randn(num_points, action_dim) * 0.1 # Add some noise back
        move_direction = (target_action - current_points)
        current_points += (1 - alpha) * move_direction * 0.1 + noise_component
        # --- !!! END OF REPLACEMENT BLOCK !!! ---
        
        # Store the state of each point at the current timestep
        for i, point in enumerate(current_points):
            trajectory_data.append({
                'x': point[0],
                'y': point[1],
                'timestep': t,
                'point_id': i
            })

    return pd.DataFrame(trajectory_data)


def app(df):
    """
    Renders the Denoising Process visualization page.
    """
    st.markdown(f"<h1 style='text-align: center;'>{NAME}</h1>", unsafe_allow_html=True)
    st.markdown("""<div style='text-align: center;'>
    This page visualizes the core mechanism of the diffusion model: **the reverse denoising process**.
    For a given state, the model starts with pure random noise and iteratively refines it over many timesteps
    to generate a final, coherent action.
    </div>""", unsafe_allow_html=True)

    # --- User Controls ---
    st.sidebar.header("Animation Controls")
    
    # In a real app, you might select a state from your dataset `df`
    # For this skeleton, we'll just use a placeholder.
    placeholder_state = np.random.rand(10) # Example state vector
    st.sidebar.info("In a real implementation, you could add a dropdown here to select different start states.")

    num_timesteps = st.sidebar.slider("Number of Denoising Timesteps (T)", 10, 100, 50)
    num_points = st.sidebar.slider("Number of Sampled Points", 50, 500, 150)

    # --- Visualization Trigger ---
    if st.button("▶️ Generate and Animate Process", type="primary"):
        
        with st.spinner("Generating denoising trajectory... This may take a moment."):
            # 1. Generate the data using our placeholder function
            trajectory_df = generate_denoising_trajectory(
                start_state=placeholder_state,
                num_timesteps=num_timesteps,
                num_points=num_points
            )
        
        st.success("Trajectory generated! Building animation...")

        # 2. Create the animated scatter plot with Plotly Express
        fig = px.scatter(
            trajectory_df,
            x='x',
            y='y',
            animation_frame='timestep',  # This is the key to the animation
            animation_group='point_id',  # This ensures points have smooth paths
            title="Denoising Trajectory from Noise to Action",
            labels={'x': 'Action Dimension 1', 'y': 'Action Dimension 2'},
            range_x=[-2, 2], # Fixed range to prevent axis jitter
            range_y=[-2, 2]
        )
        
        # Reverse the animation slider to go from T -> 0
        fig.layout.sliders[0].active = num_timesteps 
        fig.update_layout(
            height=600,
            title_font_size=20
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Observation:** Notice how the scattered cloud of points (pure noise) at the beginning of the animation
        gradually coalesces into a tight cluster. This cluster represents the final action(s) your policy has decided on.
        The path each point takes is guided by your trained model's ability to predict and remove noise at each timestep.
        """)