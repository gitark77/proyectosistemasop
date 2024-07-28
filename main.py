import streamlit as st
from app import main as app_main
from app2 import main as app2_main

def initialize_session_state():
    if 'combined_df' not in st.session_state:
        st.session_state.combined_df = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'invalid_df' not in st.session_state:
        st.session_state.invalid_df = None

def main():
    st.set_page_config(page_title="Mi Aplicación Streamlit", layout="wide")
    st.sidebar.title("Navegación")
    selection = st.sidebar.radio("Ir a", ["Extractor y validador de datos", "Analisis de datos validados"])
    
    initialize_session_state()

    if selection == "Extractor y validador de datos":
        app_main()
    elif selection == "Analisis de datos validados":
        app2_main()

if __name__ == "__main__":
    main()
