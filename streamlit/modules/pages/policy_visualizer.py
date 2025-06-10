import streamlit as st
import numpy as np
import gymnasium as gym
import torch
import tempfile
import base64
import imageio
import json
from pathlib import Path
from agents.ql_diffusion import Diffusion_QL

NAME = "Policy Visualizer"

@st.cache_resource
def load_agent_params(model_dir: str, params_path: str):
    """
    Load Diffusion_QL agent and its parameters using explicit dims from params.
    """
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
    except Exception as e:
        st.error(f"Failed to read params.json: {e}")
        return None, None, None

    env_cfg = params.get('env', {})
    state_dim = env_cfg.get('observation_space')
    action_dim = env_cfg.get('action_space')
    gym_env = params.get('gym_env_name')
    if state_dim is None or action_dim is None or gym_env is None:
        st.error("Params.json missing 'observation_space', 'action_space', or 'gym_env_name'")
        return None, None, None

    max_action = params.get('max_action', 1.0)
    device = f"cuda:{params['device_id']}" if torch.cuda.is_available() else "cpu"

    agent = Diffusion_QL(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=params.get('discount', 0.99),
        tau=params.get('tau', 0.005),
        max_q_backup=params.get('max_q_backup', False),
        beta_schedule=params.get('beta_schedule', 'linear'),
        n_timesteps=params.get('T', 5),
        eta=params.get('eta', 1.0),
        lr=params.get('lr', 0.0003),
        lr_decay=params.get('lr_decay', False),
        lr_maxt=params.get('num_epochs', 100),
        grad_norm=params.get('gn', 5.0)
    )
    try:
        agent.load_model(model_dir)
    except RuntimeError as e:
        st.error(f"Failed to load model weights: {e}")
        return None, None, None
    agent.actor.eval()
    return agent, device, params


def run_policy(env_name: str, agent, device, max_steps=1000):
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []
    total_reward = 0.0

    for _ in range(max_steps):
        state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            raw_action = agent.actor.sample(state_tensor).cpu().numpy()[0]
        # scale raw_action [-1,1] to env range
        if hasattr(env.action_space, 'high'):
            high = env.action_space.high
            low = env.action_space.low
            action = np.clip(raw_action * high, low, high)
        else:
            action = int(np.argmax(raw_action))
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(env.render())
        if terminated or truncated:
            break

    env.close()
    return frames, total_reward


def save_video(frames, path):
    imageio.mimsave(path, frames, fps=30)


def render_video(path):
    with open(path, "rb") as f:
        video_bytes = f.read()
    b64 = base64.b64encode(video_bytes).decode()
    video_html = f"""
    <video width='100%' height='auto' controls>
        <source src="data:video/mp4;base64,{b64}" type="video/mp4">
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)


def app(df=None):
    st.markdown(f"<h1 style='text-align:center;'>{NAME}</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align:center; font-size:16px; max-width:800px; margin:auto;'>
        Render your trained Diffusion-QL policy directly in its environment. Select a model run
        below, then execute and observe the resulting behavior video and total reward.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    st.sidebar.header("Model Selection")
    base_dir = st.sidebar.text_input("Base Results Directory", "results")
    runs = []
    try:
        runs = [d.name for d in Path(base_dir).iterdir() if d.is_dir()]
    except Exception:
        st.sidebar.error(f"Unable to list runs in {base_dir}")
    run_choice = st.sidebar.selectbox("Select Model Run", runs)
    model_dir = str(Path(base_dir) / run_choice)
    param_path = str(Path(model_dir) / "params.json")

    with st.spinner("Loading model and parameters..."):
        agent, device, params = load_agent_params(model_dir, param_path)
    if agent is None:
        return

    env_name = params['gym_env_name']
    max_steps = st.sidebar.slider("Max Steps", 100, 1500, 1000)

    st.markdown(
        """
        <style>
        div.stButton > button {
            font-size: 18px;
            padding: 14px 28px;
            margin: 20px auto 10px auto;
            display: block;
            background-color: #262730;
            color: white;
            border: 1px solid #444;
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button("Run Policy and Render"):
        with st.spinner("Running environment... this may take a while..."):
            try:
                frames, total_reward = run_policy(env_name, agent, device, max_steps)
            except Exception as e:
                st.error(f"Error during rollout: {e}")
                return
            tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            save_video(frames, tmp_path)
        st.success(f"Episode complete! Total reward: {total_reward:.2f}")
        render_video(tmp_path)
