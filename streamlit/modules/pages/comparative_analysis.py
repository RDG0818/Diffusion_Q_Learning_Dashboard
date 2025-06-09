# modules/pages/comparative_analysis.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# NAME attribute is required by app.py to build the navigation.
NAME = "Comparative Analysis"

def create_comparison_plot(df: pd.DataFrame):
    """
    Creates an overlaid line chart comparing the performance of multiple experiment runs.
    """
    fig = go.Figure()

    # Get a list of unique experiment names from the provided dataframe
    experiments_to_plot = df['experiment_name'].unique()

    # Use Plotly's built-in color cycle
    colors = px.colors.qualitative.Plotly

    for i, exp_name in enumerate(experiments_to_plot):
        # Filter data for the current experiment
        exp_df = df[df['experiment_name'] == exp_name]
        
        # Calculate the mean performance over steps
        # Note: Shaded min/max areas are omitted here as they become messy with many lines.
        stats_df = exp_df.groupby('step')['norm_return'].mean().reset_index()
        
        fig.add_trace(go.Scatter(
            x=stats_df['step'],
            y=stats_df['norm_return'],
            mode='lines',
            name=exp_name,
            line=dict(width=3, color=colors[i % len(colors)]),
            hovertemplate=f"<b>{exp_name}</b><br><b>Step</b>: %{{x}}<br><b>Mean Return</b>: %{{y:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        title="Performance Comparison Across Experiments",
        xaxis_title="Training Steps",
        yaxis_title="Normalized Return",
        template="plotly_dark",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )
    return fig

def create_summary_visuals(df: pd.DataFrame):
    """
    Creates a summary bar chart and a dataframe table for final metrics.
    """
    summary_data = []
    experiments_to_summarize = df['experiment_name'].unique()

    for exp_name in experiments_to_summarize:
        exp_df = df[df['experiment_name'] == exp_name]
        
        max_score = exp_df['norm_return'].max()
        
        # Calculate final score based on the average of the last 10% of steps
        last_10_percent_step = exp_df['step'].max() * 0.9
        final_score_df = exp_df[exp_df['step'] >= last_10_percent_step]
        final_score = final_score_df['norm_return'].mean()

        summary_data.append({
            "Experiment": exp_name,
            "Max Score": f"{max_score:.2f}",
            "Final Score (Avg)": f"{final_score:.2f}",
        })
    
    summary_df = pd.DataFrame(summary_data)

    # Create Bar Chart
    bar_fig = px.bar(
        summary_df,
        x="Experiment",
        y="Final Score (Avg)",
        color="Experiment",
        title="Final Score Comparison",
        template="plotly_dark",
        text="Final Score (Avg)" # Display value on top of bar
    )
    bar_fig.update_layout(showlegend=False)

    return summary_df, bar_fig

def app(df: pd.DataFrame):
    st.markdown(f"<h1 style='text-align: center;'>{NAME}</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
    Select multiple experiment runs from the sidebar to compare their performance curves and final scores side-by-side.
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # --- Data Prerequisite Check ---
    if df is None or df.empty:
        st.warning("No data loaded. Please check your data loader.")
        return
    if 'experiment_name' not in df.columns:
        st.error("Dataframe is missing the required 'experiment_name' column for comparison.")
        st.info("Please update your data loader to add an 'experiment_name' column for each run.")
        return

    # --- Sidebar Controls ---
    st.sidebar.header("Comparison Controls")
    all_experiments = sorted(df['experiment_name'].unique())
    
    # Set default selections to the first two experiments, if available
    default_selection = all_experiments[:2] if len(all_experiments) >= 2 else all_experiments

    selected_experiments = st.sidebar.multiselect(
        "Select experiments to compare:",
        options=all_experiments,
        default=default_selection
    )

    # --- Main Panel for Visualizations ---
    if not selected_experiments:
        st.info("Please select at least one experiment from the sidebar to view the analysis.")
        return

    # Filter the main dataframe based on user selection
    filtered_df = df[df['experiment_name'].isin(selected_experiments)]

    # --- Overlaid Performance Curve Plot ---
    with st.container(border=True):
        comparison_fig = create_comparison_plot(filtered_df)
        st.plotly_chart(comparison_fig, use_container_width=True)

    st.divider()

    # --- Summary Metrics Section ---
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>Quantitative Summary</h3>", unsafe_allow_html=True)
        summary_df, summary_bar_fig = create_summary_visuals(filtered_df)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("##### Key Metrics")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("##### Final Score Comparison")
            st.plotly_chart(summary_bar_fig, use_container_width=True)