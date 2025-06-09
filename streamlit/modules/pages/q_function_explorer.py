# modules/pages/q_function_explorer.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# NAME attribute is required by app.py to build the navigation.
NAME = "Q-Function Explorer"

# --- Placeholder Functions for Your Trained Models ---
# You will replace these with your actual model inference functions.

def placeholder_q_function(state: np.ndarray, action: np.ndarray) -> float:
    """
    A placeholder that simulates a trained Q-function.
    It returns a high value for actions close to a predefined "optimal" point.
    REPLACE THIS with a call to your `critic.q1(state, action)`.
    """
    optimal_action = np.array([0.6, 0.6])
    distance = np.linalg.norm(action - optimal_action)
    # Simulate a Gaussian-like peak around the optimal action
    q_value = np.exp(-distance**2 / 0.1)
    return float(q_value)

def placeholder_policy(state: np.ndarray) -> np.ndarray:
    """
    A placeholder that simulates your trained diffusion policy.
    It returns an action near the "optimal" point, with some noise.
    REPLACE THIS with a call to your action generation function.
    """
    optimal_action = np.array([0.6, 0.6])
    # Return an action slightly offset from the true optimum
    return optimal_action + np.random.randn(2) * 0.05

# --- Main Application ---

def app(df: pd.DataFrame):
    """
    Renders the Q-Function Explorer page.
    """
    st.markdown(f"<h1 style='text-align: center;'>{NAME}</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
    This page visualizes the learned Q-value landscape for a given state.
    Brighter areas indicate actions the critic estimates will lead to higher returns.
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # --- Sidebar Controls ---
    st.sidebar.header("Explorer Controls")
    
    # Use session state to hold the selected state
    if 'selected_state' not in st.session_state:
        st.session_state.selected_state = None

    if st.sidebar.button("Select a Random State from Dataset"):
        # In a real app, you would sample a state from your actual dataset `df`
        # For this skeleton, we generate a random placeholder state.
        st.session_state.selected_state = np.random.rand(10) # Assuming state dim is 10

    if st.session_state.selected_state is not None:
        st.sidebar.success("State selected! You can now generate the heatmap.")
        with st.sidebar.expander("View Selected State Vector"):
            st.write(st.session_state.selected_state)
    else:
        st.sidebar.info("Click the button above to select a random state to analyze.")

    resolution = st.sidebar.slider("Heatmap Resolution (Grid Size)", 20, 100, 40,
        help="Higher resolution is more detailed but takes longer to compute.")
    
    # --- Main Panel for Visualization ---
    if st.button("Generate Q-Value Heatmap", type="primary", use_container_width=True):
        if st.session_state.selected_state is None:
            st.error("Please select a state from the sidebar first.")
        else:
            with st.spinner(f"Generating {resolution}x{resolution} heatmap... This may take a moment."):
                # 1. Create the action grid
                action_range = np.linspace(-1.0, 1.0, resolution)
                q_values = np.zeros((resolution, resolution))

                # 2. Query the Q-function for each point on the grid
                for i, y_action in enumerate(action_range):
                    for j, x_action in enumerate(action_range):
                        action = np.array([x_action, y_action])
                        # *** REPLACE THIS with your actual Q-function call ***
                        q_values[i, j] = placeholder_q_function(st.session_state.selected_state, action)
                
                # 3. Get the action chosen by the diffusion policy
                # *** REPLACE THIS with your actual policy call ***
                policy_action = placeholder_policy(st.session_state.selected_state)

            st.success("Heatmap generated!")

            # 4. Create the Plotly figure
            fig = go.Figure(data=go.Heatmap(
                z=q_values,
                x=action_range,
                y=action_range,
                colorscale='Viridis',
                colorbar=dict(title='Q-Value')
            ))

            # 5. Overlay the policy's chosen action
            fig.add_trace(go.Scatter(
                x=[policy_action[0]],
                y=[policy_action[1]],
                mode='markers',
                marker=dict(
                    symbol='star',
                    color='red',
                    size=16,
                    line=dict(width=2, color='white')
                ),
                name='Policy Action'
            ))

            fig.update_layout(
                title="Q-Value Landscape for Selected State",
                xaxis_title="Action Dimension 1",
                yaxis_title="Action Dimension 2",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("How to Interpret This Graph"):
                st.markdown("""
                - **The Heatmap:** Represents the "value landscape" as estimated by the critic (Q-function). Brighter areas are actions the model believes will lead to higher future rewards.
                - **The Red Star:** Represents the single action that the diffusion policy chose to execute in this state.
                - **Analysis:** By comparing the star's position to the brightest areas, you can analyze your agent's behavior. Does it greedily choose the peak Q-value? Or does it select a "safer" action that is still high-value but perhaps closer to the distribution of actions seen in the training data?
                """)