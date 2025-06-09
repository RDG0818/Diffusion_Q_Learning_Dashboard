# modules/pages/state_space_explorer.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib  # To load your saved UMAP model
# You will need to install umap-learn: pip install umap-learn
# You will also need scikit-learn for joblib: pip install scikit-learn
import umap

# NAME attribute is required by app.py
NAME = "State Space Explorer"

# --- Placeholder & Model Loading Functions ---

@st.cache_resource
def load_umap_model(path='models/umap_model.joblib'):
    """Loads a pre-trained UMAP model from a file."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"UMAP model not found at '{path}'. Please train and save it first.")
        return None

def placeholder_q_function(state: np.ndarray, action: np.ndarray) -> float:
    """
    PLACEHOLDER: Simulates a trained Q-function.
    REPLACE THIS with a call to your `critic.q1(state, action)`.
    This placeholder returns a value based on the state's first component.
    """
    return float(np.sin(state[0] * np.pi) * 2)

def placeholder_policy(state: np.ndarray) -> np.ndarray:
    """
    PLACEHOLDER: Simulates your trained diffusion policy.
    REPLACE THIS with a call to your action generation function.
    """
    # Returns a dummy action of the correct dimension (e.g., 2D)
    return np.random.randn(2) * 0.1

def app(df: pd.DataFrame):
    st.markdown(f"<h1 style='text-align: center;'>{NAME}</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
    This page visualizes the agent's learned value function across a 2D projection of the state space.
    It helps answer the question: "Which regions of the state space does the agent consider valuable?"
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # --- Sidebar Controls ---
    st.sidebar.header("Explorer Controls")
    num_states_to_plot = st.sidebar.slider(
        "Number of States to Visualize", 500, 5000, 2000,
        help="More points create a denser plot but can slow down performance."
    )

    # --- Main Panel ---
    if st.button("Generate 3D Value Projection", type="primary", use_container_width=True):
        # 1. Load the UMAP model
        umap_model = load_umap_model()
        if umap_model is None:
            return

        # 2. Sample data
        if df is None or df.empty:
            st.warning("No data available to plot.")
            return
        
        # NOTE: This assumes your state vectors can be extracted from the dataframe.
        # For this skeleton, we'll generate random data.
        # REPLACE THIS with actual state extraction from your `df`.
        # For example: `sampled_states = df[state_columns].sample(num_states_to_plot).values`
        num_state_dims = 10 # Example state dimensionality
        sampled_states = np.random.rand(num_states_to_plot, num_state_dims)

        with st.spinner("Calculating values and projecting states..."):
            # 3. Calculate V(s) for each state
            # This is often the slowest part in a real implementation.
            state_values = []
            for state in sampled_states:
                # *** REPLACE a and v with your model calls ***
                action = placeholder_policy(state)
                value = placeholder_q_function(state, action)
                state_values.append(value)
            
            # 4. Project states to 2D using the loaded UMAP model
            # NOTE: In a real app, you would also load and use a StandardScaler here.
            projection_2d = umap_model.transform(sampled_states)

        st.success("Projection complete! Rendering 3D plot...")

        # 5. Create the Plotly figure
        plot_df = pd.DataFrame({
            'x': projection_2d[:, 0],
            'y': projection_2d[:, 1],
            'value': state_values
        })

        fig = go.Figure(data=[go.Scatter3d(
            x=plot_df['x'],
            y=plot_df['y'],
            z=plot_df['value'],
            mode='markers',
            marker=dict(
                size=3,
                color=plot_df['value'],  # Set color to the state value
                colorscale='Viridis',    # Choose a nice colorscale
                showscale=True,          # Display a color bar
                colorbar=dict(title='State Value V(s)'),
                opacity=0.8
            )
        )])

        fig.update_layout(
            title="3D Projection of State Space Colored by Value",
            scene=dict(
                xaxis_title='UMAP Dimension 1',
                yaxis_title='UMAP Dimension 2',
                zaxis_title='State Value V(s)'
            ),
            margin=dict(r=20, l=10, b=10, t=40),
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("How to Interpret This Graph"):
            st.markdown("""
            - **The (X, Y) Plane:** Represents the "shape" of the state space. States that are close together on this plane are similar in their original high-dimensional form. You might see distinct clusters emerge, corresponding to different types of scenarios (e.g., running forward vs. balancing).
            - **The Z-Axis (Height) & Color:** Represents the value `V(s)` of each state, as estimated by your agent. Higher, brighter points are states the agent considers more valuable and likely to lead to higher rewards.
            - **Analysis:** This plot can reveal "islands" of high-value states. It helps you understand if your agent has learned to value specific, isolated regions of the state space or if it has learned a more generalized value function.
            """)