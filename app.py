import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
import pandas as pd
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TrialSight - Criteria Extractor",
    page_icon="💊",
    layout="centered"
)

# --- AESTHETICS ---
st.markdown("""
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    .main .block-container { padding: 2rem; padding-bottom: 5rem; max-width: 800px; }
    h1, h2, h3 { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# --- API KEY SETUP ---
# This will be configured in your secrets.toml file
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("🚨 Google API Key not found. Please create a .streamlit/secrets.toml file with your GOOGLE_API_KEY.")
    st.stop()

# --- UI & APP LOGIC ---
st.title("💊 TrialSight: Criteria Extractor")
st.markdown("**The Mission:** Upload clinical trial PDFs and instantly extract the inclusion and exclusion criteria into a clean, downloadable table.")

uploaded_files = st.file_uploader(
    "Upload one or more clinical trial PDFs", 
    type="pdf", 
    accept_multiple_files=True
)

if st.button("Extract Criteria", type="primary", use_container_width=True):
    if uploaded_files:
        st.info("This is the skeleton of our MVP. The AI extraction engine will be built here next, after we validate the exact pain point with researchers.")
        # The core AI logic will be implemented in our next development sprint.
    else:
        st.warning("Please upload at least one PDF file to begin.")
