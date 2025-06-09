# app.py
import streamlit as st
import importlib
from modules.data_loader import load_all_experiments
import modules.pages as pages_pkg

# Dynamically load every page module listed in modules/pages/__init__.py
PAGE_FUNCS = {}
for module_name in pages_pkg.__all__:
    module = importlib.import_module(f"modules.pages.{module_name}")
    PAGE_FUNCS[module.NAME] = module.app


def main():
    st.set_page_config(page_title="Diffusion Q-L Dashboard", layout="wide")
    st.sidebar.title("Navigation")
    
    # ... (your df = load_all_seeds(...) call)
    df = load_all_experiments("results")

    # Sidebar selector
    page = st.sidebar.selectbox("Go to", list(PAGE_FUNCS.keys()))

    PAGE_FUNCS[page](df)

if __name__ == "__main__":
    main()