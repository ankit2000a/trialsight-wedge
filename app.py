import streamlit as st
import fitz  # PyMuPDF
# import difflib # No longer needed for HTML diff
import os
import time
import re
import unicodedata # For robust normalization
import diff_match_patch as dmp_module # Use DMP for visual diff now

# --- Page Configuration ---
st.set_page_config(
    page_title="TrialSight: Document Comparator",
    page_icon="🔎",
    layout="wide"
)

# --- App Styling (CSS) ---
# Updated CSS for DMP output
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E2E8F0; }
    .stButton>button { border-radius: 0.5rem; }
    .stFileUploader { border: 2px dashed #4A5568; background-color: #1A202C; border-radius: 0.5rem; padding: 2rem; text-align: center; }
    .file-card { background-color: #2D3748; border-radius: 0.5rem; padding: 1rem; border: 1px solid #4A5568; display: flex; justify-content: space-between; align-items: center; }

    /* Styles for diff-match-patch HTML output */
    .diff-container {
        font-family: Consolas, 'Courier New', monospace;
        font-size: 0.9em;
        line-height: 1.4;
        border: 1px solid #4A5568;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #1A202C; /* Match uploader background */
        overflow-x: auto; /* Allow horizontal scroll if needed */
        white-space: pre-wrap; /* Preserve whitespace and wrap */
        word-wrap: break-word;
    }
    ins { /* Insertions */
        background-color: rgba(16, 185, 129, 0.2); /* Lighter Green background */
        color: #A7F3D0; /* Green text */
        text-decoration: none;
    }
    del { /* Deletions */
        background-color: rgba(239, 68, 68, 0.2); /* Lighter Red background */
        color: #FECACA; /* Red text */
        text-decoration: none; /* Remove strikethrough for background highlight */
        /* Optional: Add subtle underline */
        /* border-bottom: 1px dotted rgba(239, 68, 68, 0.6); */
    }

    /* Loader */
    .loader-container { display: flex; justify-content: center; align-items: center; flex-direction: column; margin: 20px 0; }
    .loader { border: 4px solid #4b5563; border-left-color: #3b82f6; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
def init_session_state():
    # Simplified state
    keys = ['file1_data', 'file2_data', 'diff_html_output', # Renamed state key
            'original_text_normalized', 'revised_text_normalized',
            'processing_comparison']
    for key in keys:
        if key not in st.session_state: st.session_state[key] = None
    if st.session_state.processing_comparison is None: st.session_state.processing_comparison = False
init_session_state()

# --- NO AI CONFIGURATION NEEDED FOR THIS VERSION ---

# --- Helper Functions ---

# --- USING CHATGPT'S NORMALIZATION FUNCTION ---
@st.cache_data
def normalize_pdf_text(file_bytes, filename="file"):
    """
    Extracts and normalizes text preserving line structure, joining broken words.
    Based on user-provided ChatGPT suggestion.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text", sort=True) for page in doc)
        if not text: return f"ERROR: No text found in PDF {filename}"
        # Normalization steps...
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'([a-z])\n([a-z])', r'\1\2', text, flags=re.IGNORECASE)
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = text.split('\n')
        stripped_lines = [line.strip() for line in lines]
        non_blank_lines = [line for line in stripped_lines if line]
        text = '\n'.join(non_blank_lines)
        text = text.strip()
        # --- REMOVED LOWERCASE STEP FOR VISUAL DIFF ---
        # text = text.lower() # Keep case for visual diff
        if not text: return f"ERROR: Text became empty after normalization for {filename}"
        return text
    except Exception as e:
        print(f"Error normalizing {filename}: {e}")
        return f"ERROR: Could not read/normalize {filename}. Details: {e}"

# --- NEW FUNCTION using diff-match-patch for HTML ---
def generate_dmp_diff_html(text1_norm, text2_norm):
    """Creates highlighted HTML diff using diff-match-patch."""
    if text1_norm is None or text2_norm is None or text1_norm.startswith("ERROR:") or text2_norm.startswith("ERROR:"):
         return "<p style='color:red;'>Error: Cannot generate visual diff (text normalization failed).</p>"
    try:
        dmp = dmp_module.diff_match_patch()
        dmp.Diff_Timeout = 2.0 # Increase timeout slightly for potentially complex diffs
        diffs = dmp.diff_main(text1_norm, text2_norm)
        # cleanupSemantic can sometimes merge small changes, might be useful
        dmp.diff_cleanupSemantic(diffs)
        # cleanupEfficiency optimizes diffs further, might reduce noise slightly
        dmp.diff_cleanupEfficiency(diffs)

        html = dmp.diff_prettyHtml(diffs)
        # Wrap in a container div for styling and overflow control
        return f"<div class='diff-container'>{html}</div>"
    except Exception as e:
        print(f"Error generating DMP HTML diff: {e}")
        # Return error message formatted as HTML
        return f"<p style='color:red;'>Error: Failed to generate visual comparison using diff-match-patch. {e}</p>"


# --- Main App UI ---
st.title("📄 TrialSight: Document Content Comparator") # Updated title slightly
st.markdown("Highlights **content changes** between two PDFs, ignoring most formatting differences.") # Updated description
st.markdown("---")

# File Uploader & Display Logic
if not st.session_state.get('file1_data') and not st.session_state.get('file2_data'):
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, key="file_uploader")
    if uploaded_files:
        if len(uploaded_files) != 2: st.warning("Please upload exactly two files.")
        else:
            st.session_state.file1_data = uploaded_files[0]
            st.session_state.file2_data = uploaded_files[1]
            # Clear all results on new upload
            keys_to_clear = ['diff_html_output', 'summary', 'original_text_normalized', 'revised_text_normalized', 'processing_comparison']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            st.session_state.processing_comparison = False
            st.rerun()
else:
    col1, col2 = st.columns(2)
    file1_name = st.session_state.file1_data.name if st.session_state.get('file1_data') else "N/A"
    file2_name = st.session_state.file2_data.name if st.session_state.get('file2_data') else "N/A"
    with col1: st.success(f"Original: **{file1_name}**")
    with col2: st.success(f"Revised: **{file2_name}**")
    if st.button("Clear Files and Start Over"):
        keys_to_clear = ['file1_data', 'file2_data', 'diff_html_output', 'summary', 'original_text_normalized', 'revised_text_normalized', 'processing_comparison']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

# --- Comparison Logic ---
if st.session_state.get('file1_data') and st.session_state.get('file2_data'):
    # Show Compare button only if needed
    if not st.session_state.get('diff_html_output') and not st.session_state.get('processing_comparison'):
        if st.button("Compare Documents", type="primary", use_container_width=True):
            st.session_state.processing_comparison = True
            # Clear previous results
            keys_to_clear = ['diff_html_output', 'summary', 'original_text_normalized', 'revised_text_normalized']
            for key in keys_to_clear:
                 if key in st.session_state: del st.session_state[key]
            st.rerun()

    # Execute comparison if flagged
    if st.session_state.get('processing_comparison'):
        with st.spinner("Reading, normalizing, and comparing documents..."):
            time.sleep(0.5) # Ensure spinner shows
            file1 = st.session_state.file1_data
            file2 = st.session_state.file2_data
            if file1 and file2:
                file1_bytes = file1.getvalue()
                file2_bytes = file2.getvalue()

                # --- Extract NORMALIZED text using the CORRECT function ---
                text1_norm = normalize_pdf_text(file1_bytes, file1.name)
                text2_norm = normalize_pdf_text(file2_bytes, file2.name)

                # Store normalized text
                st.session_state['original_text_normalized'] = text1_norm
                st.session_state['revised_text_normalized'] = text2_norm

                # Generate HTML diff using DMP (handles internal errors)
                st.session_state['diff_html_output'] = generate_dmp_diff_html(text1_norm, text2_norm)
            else:
                st.session_state['diff_html_output'] = "<p style='color:red;'>Error: File data missing.</p>" # Error as HTML
                st.session_state['summary'] = None # Ensure summary is cleared too
            st.session_state.processing_comparison = False # Reset flag
            # Rerun regardless of diff success to show either diff or error
            st.rerun()


# --- Display Results Section ---
# Display diff if processing is done and diff_html_output exists
if not st.session_state.get('processing_comparison') and st.session_state.get('diff_html_output'):

    # --- DEBUGGER (Shows the robustly normalized text) ---
    with st.expander("Show/Hide Normalized Text (Used for Comparison)"):
        col1, col2 = st.columns(2)
        original_display = st.session_state.get('original_text_normalized', "Not available")
        revised_display = st.session_state.get('revised_text_normalized', "Not available")
        # Display potentially long text in text_area for scrollability
        with col1: st.subheader("Original (Normalized)"); st.text_area("Original Norm", original_display, height=200, key="dbg_txt1_norm")
        with col2: st.subheader("Revised (Normalized)"); st.text_area("Revised Norm", revised_display, height=200, key="dbg_txt2_norm")


    # --- Visual Diff using diff-match-patch output ---
    st.subheader("Visual Comparison of Content Changes")
    st.markdown("*(Green = Added, Red Background = Deleted. Formatting differences are ignored)*") # Updated note
    # Use st.markdown with unsafe_allow_html=True for the DMP output
    st.markdown(st.session_state.diff_html_output, unsafe_allow_html=True)

    # --- AI Summary Section REMOVED ---
    # st.markdown("---")
    # st.subheader("🤖 AI-Powered Summary of Content Changes")
    # ... (All AI summary button and display logic removed) ...


# Handle Errors / Loading State
elif st.session_state.get('processing_comparison'):
     st.markdown('<div class="loader-container"><div class="loader"></div><p>Processing...</p></div>', unsafe_allow_html=True)
# Display error if diff generation failed (now diff_html_output contains the HTML error)
elif not st.session_state.get('processing_comparison') and st.session_state.get('diff_html_output'):
     st.markdown(st.session_state.diff_html_output, unsafe_allow_html=True)

