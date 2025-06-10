import streamlit as st
import importlib
from modules.data_loader import load_old_experiments, load_new_experiments
import modules.pages as pages_pkg

# Dynamically load every page module listed in modules/pages/__init__.py
def _load_pages():
    PAGE_FUNCS = {}
    for module_name in pages_pkg.__all__:
        module = importlib.import_module(f"modules.pages.{module_name}")
        PAGE_FUNCS[module.NAME] = module.app
    return PAGE_FUNCS


def main():
    st.set_page_config(page_title="Diffusion Q-L Dashboard", layout="wide")
    st.sidebar.title("Navigation")

    old_df = load_old_experiments("streamlit/orig_results")
    new_df = load_new_experiments("results")

    PAGE_FUNCS = _load_pages()

    page = st.sidebar.selectbox("Go to", list(PAGE_FUNCS.keys()))

    if page == "Home":
        df = old_df
    else:
        df = new_df

    PAGE_FUNCS[page](df)


if __name__ == "__main__":
    main()
