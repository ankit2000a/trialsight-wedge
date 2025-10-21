import streamlit as st
import fitz  # PyMuPDF
import difflib
import google.generativeai as genai
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="TrialSight: Document Comparator",
    page_icon="🔎",
    layout="wide"
)

# --- App Styling (CSS) ---
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #E2E8F0;
    }
    .stButton>button {
        border-radius: 0.5rem;
    }
    .stFileUploader {
        border: 2px dashed #4A5568;
        background-color: #1A202C;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
    }
    .file-card {
        background-color: #2D3748;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #4A5568;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .diff-ins {
        background-color: rgba(16, 185, 129, 0.2);
        color: #6EE7B7;
        text-decoration: none;
    }
    .diff-del {
        background-color: rgba(239, 68, 68, 0.2);
        color: #FCA5A5;
        text-decoration: line-through;
    }
    .loader-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        margin: 20px 0;
    }
    .loader {
        border: 4px solid #4b5563;
        border-left-color: #3b82f6;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .summary-box {
        background-color: #1A202C;
        border: 1px solid #4A5568;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
def init_session_state():
    for key in ['files', 'diff_html', 'summary', 'original_text', 'revised_text']:
        if key not in st.session_state:
            st.session_state[key] = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

init_session_state()

# --- Gemini API Configuration ---
ai_enabled = False
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro') 
        ai_enabled = True
    else:
        st.warning("Google API Key not found. The AI Summary feature is disabled.")

except Exception as e:
    ai_enabled = False
    st.warning(f"Could not initialize Google AI: {e}")


# --- Helper Functions ---

# ***FIX 1:***
# We cache the function that takes BYTES, not the file object.
# This makes the cache reliable.
@st.cache_data
def extract_text_from_bytes(file_bytes, filename="file"):
    """Extracts text from PDF bytes."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = " ".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        return f"Error reading {filename}: {e}"

def generate_diff_html(text1, text2, filename1="Original", filename2="Revised"):
    """Creates a side-by-side HTML diff of two texts."""
    d = difflib.HtmlDiff(wrapcolumn=80)
    html = d.make_table(text1.splitlines(), text2.splitlines(), fromdesc=filename1, todesc=filename2)
    style = """
    <style>
    table.diff { font-family: monospace; border-collapse: collapse; width: 100%; }
    .diff_header { background-color: #374151; color: #E5E7EB; }
    .diff_add { background-color: #052e16; }
    .diff_chg { background-color: #4d380c; }
    .diff_sub { background-color: #4c1d1d; }
    </style>
    """
    return style + html

def get_ai_summary(text1, text2):
    """Generates a summary of changes using the Gemini API."""
    if not ai_enabled:
        return "AI Summary feature is not available."

    diff = list(difflib.unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromfile='Original',
        tofile='Revised',
    ))
    diff_text = "".join([line for line in diff if line.startswith(('+', '-')) and not line.startswith(('---', '+++'))])

    if not diff_text.strip():
        return "No significant textual differences were found to summarize."

    prompt = f"""
    You are an expert clinical trial protocol reviewer. Analyze the following changes between two versions of a document and generate a concise, bulleted list of the most significant modifications.
    Focus specifically on substantive changes related to:
    1.  Inclusion/Exclusion criteria
    2.  Dosage information or treatment schedules
    3.  Study procedures or assessments
    4.  Safety reporting requirements
    5.  Key objectives or endpoints
    Ignore minor grammatical corrections, formatting changes, or rephrasing unless it alters the meaning. Structure your output clearly using markdown.

    Here are the extracted changes (lines starting with '+' were added, lines with '-' were removed):
    ---
    {diff_text[:8000]}
    ---
    **Summary of Key Changes:**
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with the AI model: {e}"

# --- Main App UI ---
st.title("📄 TrialSight: Document Comparator")
st.markdown("Compare two versions of a document to see changes and get an AI-powered summary.")
st.markdown("---")

# File Uploader
if not st.session_state.get('file1_data') and not st.session_state.get('file2_data'):
    uploaded_files = st.file_uploader(
        "Upload the original and revised PDF files to compare.",
        type="pdf",
        accept_multiple_files=True,
        key="file_uploader"
    )
    if uploaded_files:
        if len(uploaded_files) != 2:
            st.warning("Please upload exactly two files.")
        else:
            st.session_state.file1_data = uploaded_files[0]
            st.session_state.file2_data = uploaded_files[1]
            st.rerun()
else:
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"Original: **{st.session_state.file1_data.name}**")
    with col2:
        st.success(f"Revised: **{st.session_state.file2_data.name}**")

    if st.button("Clear Files and Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Main Logic ---
if st.session_state.get('file1_data') and st.session_state.get('file2_data'):
    if st.button("Compare Documents", type="primary", use_container_width=True):
        st.session_state.processing_comparison = True
        st.session_state.diff_html = None
        st.session_state.summary = None

        with st.spinner("Reading and comparing documents..."):
            file1 = st.session_state.file1_data
            file2 = st.session_state.file2_data
            
            # ***FIX 2:***
            # We read the bytes from the files first.
            file1_bytes = file1.getvalue()
            file2_bytes = file2.getvalue()
            
            # Then we pass the BYTES to our cached function.
            text1 = extract_text_from_bytes(file1_bytes, file1.name)
            text2 = extract_text_from_bytes(file2_bytes, file2.name)
            
            if text1 is not None and text2 is not None:
                st.session_state['original_text'] = text1
                st.session_state['revised_text'] = text2
                st.session_state['diff_html'] = generate_diff_html(text1, text2, file1.name, file2.name)
            else:
                st.error("Failed to process one or both PDF files.")
            st.session_state.processing_comparison = False

# --- Display Results Section ---
if st.session_state.get('diff_html'):
    
    # --- DEBUGGER (Still useful!) ---
    with st.expander("Show/Hide Extracted Text (For Debugging)"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Text from Original")
            st.text_area("Original Text", st.session_state.original_text, height=200, key="debug_text1")
        with col2:
            st.subheader("Text from Revised")
            st.text_area("Revised Text", st.session_state.revised_text, height=200, key="debug_text2")
    # --- END DEBUGGER ---

    with st.expander("Show/Hide Side-by-Side Diff", expanded=True):
        st.components.v1.html(st.session_state.diff_html, height=600, scrolling=True)

    st.markdown("---")
    st.subheader("🤖 AI-Powered Summary")
    st.markdown("Click the button below for a summary of the key changes identified in the documents.")

    if st.button("✨ Get Summary", use_container_width=True, disabled=not ai_enabled, key="generate_summary"):
        with st.spinner("Analyzing changes with Gemini AI... This might take a moment."):
            summary = get_ai_summary(st.session_state.original_text, st.session_state.revised_text)
            st.session_state['summary'] = summary
            
if st.session_state.get('summary'):
    st.markdown("### Summary of Changes:")
    st.markdown(st.session_state.summary)