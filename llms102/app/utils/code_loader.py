import streamlit as st

def show_source(filepath):
    with open(filepath, "r") as f:
        st.code(f.read(), language="python")

def run_it_yourself(code: str):
    st.subheader("Run this yourself")
    st.caption("Clone the repo, pip install -e ., then run:")
    st.code(code, language="python")