# modules/pages/model_interpretability.py
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# NAME attribute is required by app.py to build the navigation.
NAME = "Model Interpretability (XAI)"

# --- Placeholder Functions for Your Trained Model ---
# You will replace these with your actual model and data.

def placeholder_q_function(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """
    A placeholder that simulates a trained Q-function.
    It must accept a batch of states and actions and return a batch of Q-values.
    REPLACE THIS with a call to your `critic.q1(states, actions)`.

    This placeholder logic creates a simple, explainable Q-value.
    """
    # Simple weighted sum of state features, plus a bonus for a specific action
    state_weights = np.array([0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.5, 0.1, 0.9, -0.7])
    action_bonus = (actions[:, 0] > 0.5) * 0.5  # Bonus if first action dim is high
    
    q_values = np.dot(states, state_weights) + action_bonus
    return q_values

def app(df: pd.DataFrame):
    st.markdown(f"<h1 style='text-align: center;'>{NAME}</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
    This page uses SHAP (SHapley Additive exPlanations) to explain *why* the model assigns a certain Q-value to a state-action pair.
    It shows how each feature of the state contributes to the final prediction.
    </div>
    """, unsafe_allow_html=True)
    st.info("Note: SHAP calculations can be computationally intensive and may take a moment to run.", icon="⚙️")
    st.divider()

    # --- Sidebar Controls & Setup ---
    st.sidebar.header("XAI Controls")

    # Use session state to hold the selected instance
    if 'instance_to_explain' not in st.session_state:
        st.session_state.instance_to_explain = None

    if st.sidebar.button("Select a Sample Instance to Explain"):
        # In a real app, you might provide a dropdown of interesting states.
        # For this skeleton, we generate a random state and action.
        num_state_dims = 10
        st.session_state.instance_to_explain = {
            "state": np.random.rand(1, num_state_dims),
            "action": np.random.rand(1, 2) # Assuming 2D action
        }

    if st.session_state.instance_to_explain:
        st.sidebar.success("Instance selected!")
        with st.sidebar.expander("View Selected State & Action"):
            st.write("**State Vector:**", st.session_state.instance_to_explain['state'])
            st.write("**Action Vector:**", st.session_state.instance_to_explain['action'])
    else:
        st.sidebar.info("Click the button above to select a sample state-action pair.")
        return

    # --- Main Panel for SHAP Analysis ---
    if st.button("Generate SHAP Explanation", type="primary", use_container_width=True):
        instance = st.session_state.instance_to_explain
        selected_state = instance['state']
        selected_action = instance['action']

        with st.spinner("Setting up SHAP explainer and calculating values..."):
            # 1. Define the prediction function for SHAP.
            # SHAP's KernelExplainer needs a function that accepts a numpy array of states
            # and returns a numpy array of model outputs (Q-values).
            def q_function_for_shap(states_array):
                # We fix the action and explain the Q-value based on the state features.
                num_samples = states_array.shape[0]
                actions_array = np.tile(selected_action, (num_samples, 1))
                # *** REPLACE THIS with your actual Q-function call ***
                return placeholder_q_function(states_array, actions_array)

            # 2. Create a "background" dataset for the explainer's expectations.
            # This should be a representative sample of states from your dataset.
            # *** REPLACE THIS with a sample from your actual data ***
            background_states = np.random.rand(100, selected_state.shape[1])

            # 3. Create the SHAP Explainer
            explainer = shap.KernelExplainer(q_function_for_shap, background_states)

            # 4. Calculate SHAP values for the single instance we want to explain
            shap_values = explainer.shap_values(selected_state)
            
            # For plotting, create placeholder feature names
            feature_names = [f'State_Feature_{i}' for i in range(selected_state.shape[1])]

        st.success("SHAP explanation generated!")

        # --- Display SHAP Plots ---
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>SHAP Force Plot</h3>", unsafe_allow_html=True)
            st.markdown("This plot shows features that push the prediction higher (red) or lower (blue).")
            # Use st.components.v1.html to render the JS-based force plot
            shap.initjs()
            force_plot_html = shap.force_plot(
                explainer.expected_value, shap_values, selected_state, feature_names=feature_names, show=False
            )
            st.components.v1.html(force_plot_html.html(), height=100)
        
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>SHAP Waterfall Plot</h3>", unsafe_allow_html=True)
            st.markdown("This plot shows how each feature's impact builds upon the base value to reach the final prediction.")
            
            # Create the waterfall plot object
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=selected_state[0],
                feature_names=feature_names
            ), show=False)
            plt.tight_layout()
            st.pyplot(fig)