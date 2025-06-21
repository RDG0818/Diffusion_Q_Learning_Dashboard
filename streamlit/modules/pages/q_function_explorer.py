import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import minari
import os
import json
from agents.ql_diffusion import Diffusion_QL

NAME = "Q-Function Explorer"

@st.cache_resource
def load_agent(model_dir: str):
    """
    Loads a saved Diffusion_QL agent using the params.json from its directory.
    """
    print(f"Loading trained agent from: {model_dir}")
    params_path = os.path.join(model_dir, "params.json")
    if not os.path.exists(params_path):
        st.error(f"params.json not found in {model_dir}")
        return None, None

    with open(params_path, 'r') as f:
        params = json.load(f)
    
    device = f"cuda:{params.get('device_id', 0)}" if torch.cuda.is_available() else "cpu"
    
    try:
        agent = Diffusion_QL(
            state_dim=params['env']['observation_space'],
            action_dim=params['env']['action_space'],
            max_action=params.get('max_action', 1.0),
            device=device,
            discount=params['discount'], tau=params['tau'], max_q_backup=params['max_q_backup'],
            beta_schedule=params['beta_schedule'], n_timesteps=params['T'], eta=params['eta'],
            lr=params['lr'], lr_decay=params['lr_decay'], lr_maxt=params['num_epochs'], grad_norm=params['gn']
        )
        agent.load_model(model_dir)
        agent.actor.eval(); agent.critic.eval()
        return agent, device, params
    except Exception as e:
        st.error(f"Failed to load agent. Error: {e}")
        return None, None, None

@st.cache_data
def load_states_from_file(env_name: str):
    """
    Loads a NumPy array of states from the 'states/' directory.
    """
    states_path = f"states/{env_name}.npy"
    print(f"Loading states from: {states_path}")
    if not os.path.exists(states_path):
        st.error(f"State file not found at '{states_path}'. Please ensure it exists.")
        return None
    return np.load(states_path)

def compute_q_grid(agent, state: torch.Tensor, dim1: int, dim2: int, resolution: int = 40):
    """
    Compute a grid of Q-values for two action dims (others zero).
    """
    action_dim = agent.action_dim
    vals = np.linspace(-1.0, 1.0, resolution)
    X, Y = np.meshgrid(vals, vals)
    batch = np.zeros((resolution*resolution, action_dim), dtype=np.float32)
    batch[:, dim1] = X.flatten()
    batch[:, dim2] = Y.flatten()
    actions = torch.from_numpy(batch).to(state.device)
    states = state.repeat(resolution*resolution, 1)
    with torch.no_grad():
        Q = agent.critic.q_min(states, actions).cpu().numpy().reshape(resolution, resolution)
    return Q, vals


def app(df=None):
    # Header and description
    st.markdown(f"<h1 style='text-align:center;'>{NAME}</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align:center; font-size:16px; max-width:800px; margin:auto;'>
        This page dissects the learned action-value function, or <strong>Q-function</strong>, which is the critic's estimate of the total discounted reward from taking an action <em>a</em> in a state <em>s</em>.
        The visualizations below reveal the Q-function's structure for a single, chosen state. By sweeping two action dimensions across their range while holding others at zero, we can create a "value landscape" to see which actions the critic has learned to prefer.
        Each violin plot below summarizes the distribution of all Q-values from the corresponding heatmap above. The shape illustrates the density of values—wider sections indicate that more actions in the grid yield that Q-value. The horizontal cyan line marks the value of the action chosen by the policy, showing how its decision ranks against all other possibilities.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("<h3 style='text-align: center;'>Q-Function Training Objective</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align:center; font-size:16px; max-width:800px; margin:auto;'>
        The Q-networks are trained in a conventional way by minimizing the Bellman error, using tricks from modern RL like Double Q-Learning to stabilize training. The objective is to make the Q-value of the current state and action as close as possible to the reward received plus the discounted value of the next state and action, as determined by the target networks. The full objective is:
        </div>
        """,
        unsafe_allow_html=True
    )
    st.latex(r'''
    \mathcal{L}(\phi_i) = \mathbb{E}_{(s_t, a_t, s_{t+1}) \sim \mathcal{D}} \left[ \left\| \left(r(s_t, a_t) + \gamma \min_{k=1,2} Q_{\phi_k'}(s_{t+1}, a_{t+1}^0)\right) - Q_{\phi_i}(s_t, a_t) \right\|^2 \right]
    ''')

    st.divider()
    # Model & data
    st.sidebar.header("Experiment Selection")

    # --- NEW: Dropdown for Model Directory ---
    results_dir = "results"
    if not os.path.exists(results_dir):
        st.error(f"Results directory not found at '{results_dir}'")
        return
    
    # Get all subdirectories in the results folder
    experiment_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    if not experiment_dirs:
        st.error(f"No experiment folders found in '{results_dir}'")
        return

    selected_model_dir = st.sidebar.selectbox("Select Model Directory", experiment_dirs)
    MODEL_DIR = os.path.join(results_dir, selected_model_dir)
    # --- END NEW ---

    # Load agent and params
    agent, device, params = load_agent(MODEL_DIR)
    if agent is None:
        return

    # --- NEW: Load states from the corresponding file ---
    env_name_from_params = params.get('env', {}).get('name')
    if not env_name_from_params:
        st.error("Environment name not found in params.json.")
        return
        
    observations = load_states_from_file(env_name_from_params)
    if observations is None:
        return
    # --- END NEW ---

    # State selection now uses the loaded observations
    st.sidebar.header("State Selection")
    if 'state_idx' not in st.session_state:
        st.session_state.state_idx = 0
        
    if st.sidebar.button("Select Random State"):
        st.session_state.state_idx = np.random.randint(len(observations))
    
    idx = st.sidebar.number_input("State Index", 0, len(observations)-1, st.session_state.state_idx)
    st.session_state.state_idx = idx
    state = torch.from_numpy(observations[idx]).unsqueeze(0).to(device)
    st.sidebar.write(f"Displaying state #{idx} from `{env_name_from_params}.npy`")

    # Dimension pairs (now dynamic based on action_dim)
    st.sidebar.header("Action Dim Pairs")
    dims = list(range(agent.action_dim))
    pairs = [(i, j) for i in dims for j in dims if i < j]
    # default to all pairs if ≤4, otherwise first 4
    default = pairs if len(pairs) <= 4 else pairs[:4]
    pairs = st.sidebar.multiselect("Pick pairs", pairs, default)

    # Policy action
    with torch.no_grad():
        a_star = agent.actor.sample(state).cpu().numpy()[0]
        q_star = agent.critic.q_min(state, torch.from_numpy(a_star.astype(np.float32)).unsqueeze(0).to(device)).cpu().item()

    # Button style
    st.markdown("""
    <style>
    div.stButton > button {font-size:20px; padding:16px 40px; margin:auto; display:block;}
    div.stSidebar div.stButton > button {font-size:18px; padding:14px 30px;}
    </style>
    """, unsafe_allow_html=True)

    # Landscapes title
    st.markdown("<h2 style='text-align:center; margin-top:0;'>Q-Value Landscapes</h2>", unsafe_allow_html=True)
    if st.button("Generate Q-Value Landscapes"):
        if not pairs:
            st.warning("Select at least one pair.")
            return
        # Heatmap grid
        n = len(pairs)
        cols = min(3, n); rows = (n+cols-1)//cols
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Dims {i} vs {j}" for i,j in pairs], specs=[[{'type':'heatmap'}]*cols for _ in range(rows)])
        for k,(i,j) in enumerate(pairs):
            r,c = divmod(k, cols); r+=1; c+=1
            Q, vals = compute_q_grid(agent, state, i, j)
            fig.add_trace(go.Heatmap(z=Q, x=vals, y=vals, colorscale='Viridis', showscale=(c==cols)), row=r, col=c)
            fig.add_trace(go.Scatter(x=[a_star[i]], y=[a_star[j]], mode='markers', marker=dict(symbol='star', size=14, color='red', line=dict(width=2, color='white'))), row=r, col=c)
        fig.update_layout(height=300*rows, width=300*cols, showlegend=False, margin=dict(t=30,l=20,r=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Violin distributions
        st.markdown("<h2 style='text-align:center;'>Q-Value Distributions per Pair</h2>", unsafe_allow_html=True)
        data=[]
        for (i,j) in pairs:
            Q,_=compute_q_grid(agent, state, i, j)
            for v in Q.flatten(): data.append({'pair':f"{i},{j}",'q':v})
        import pandas as pd
        df_q=pd.DataFrame(data)
        viol=go.Figure()
        viol.update_layout(xaxis={'type': 'category'})
        for p in df_q['pair'].unique():
            viol.add_trace(go.Violin(x=df_q[df_q['pair']==p]['pair'], y=df_q[df_q['pair']==p]['q'], name=p, box_visible=True, meanline_visible=True))
        viol.add_hline(y=q_star, line_color='cyan', line_dash='dot', annotation_text='Policy Q', annotation_position='top right')
        viol.update_layout(yaxis_title='Q-value', xaxis_title='Pair', height=400, margin=dict(t=40,b=20))
        st.plotly_chart(viol, use_container_width=True)
