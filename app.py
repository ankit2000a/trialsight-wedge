import streamlit as st
import fitz
import difflib
import google.generativeai as genai
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="TrialSight: Document Comparator",
    page_icon="🔎",
    layout="wide"
)

# --- App Styling (CSS) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E2E8F0; }
    h1 { color: #FFFFFF; }
    .stButton>button { border-radius: 0.5rem; }
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
    }
    .diff-ins { background-color: rgba(16, 185, 129, 0.2); color: #6EE7B7; text-decoration: none; }
    .diff-del { background-color: rgba(239, 68, 68, 0.2); color: #FCA5A5; text-decoration: line-through; }
    .loader {
        border: 4px solid #4b5563;
        border-top: 4px solid #3b82f6;
        border-radius: 50%;
        width: 40px; height: 40px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
""", unsafe_allow_html=True)

# --- Gemini API Configuration ---
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.secrets.get("GOOGLE_API_KEY")

    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        ai_enabled = True
    else:
        ai_enabled = False
        st.warning("Google API Key not found. The AI Summary feature will be disabled. To enable it, set up your secrets or environment variable named 'GOOGLE_API_KEY'.")

except Exception as e:
    ai_enabled = False
    st.warning(f"Could not initialize Google AI. Error: {e}")

# --- Helper Functions ---
@st.cache_data
def read_pdf_text(uploaded_file):
    try:
        file_bytes = uploaded_file.getvalue()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = " ".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return None

def generate_diff_html(text1, text2, filename1="Original", filename2="Revised"):
    d = difflib.HtmlDiff(wrapcolumn=80)
    html = d.make_table(text1.splitlines(), text2.splitlines(), fromdesc=filename1, todesc=filename2)
    style = """
    <style>
        table.diff {
            font-family: monospace;
            border-collapse: collapse;
            width: 100%;
        }
        .diff_header { background-color: #374151; color: #E5E7EB; }
        .diff_add { background-color: #052e16; }
        .diff_chg { background-color: #4a2100; }
        .diff_sub { background-color: #4c1d1d; }
    </style>
    """
    return style + html

def get_ai_summary(text1, text2):
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
        return "No significant textual differences found to summarize."

    prompt = f"""
    You are an expert clinical trial protocol reviewer. Analyze the following changes between two versions of a document and generate a concise, bulleted list summarizing only the most significant modifications.

    Focus specifically on substantive changes related to:
    1.  Inclusion/Exclusion criteria
    2.  Dosage information, treatment schedules, or study procedures
    3.  Safety reporting requirements
    4.  Key objectives or endpoints
    5.  Any changes in numerical values (e.g., age ranges, lab values).

    Ignore minor grammatical corrections, formatting changes, or rephrasing unless it alters the meaning.

    CHANGES:
    ```
    {diff_text[:8000]}
    ```

    Provide your summary in clear, easy-to-read markdown format.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error communicating with the AI model: {e}")
        return "Could not generate summary due to an API error."

# --- Main App Logic ---
st.title("TrialSight: Protocol Comparator")
st.markdown("Compare two versions of a clinical trial protocol to see what's changed and get an AI-powered summary.")

uploaded_files = st.file_uploader(
    "Upload two PDF files for comparison",
    type="pdf",
    accept_multiple_files=True,
    key="file_uploader"
)

if uploaded_files and len(uploaded_files) == 2:
    st.session_state['file1_data'] = uploaded_files[0]
    st.session_state['file2_data'] = uploaded_files[1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"Original: **{st.session_state.file1_data.name}**")
    with col2:
        st.success(f"Revised: **{st.session_state.file2_data.name}**")

elif uploaded_files:
    st.warning("Please upload exactly two files to compare.")
    st.session_state.clear() # Clear state if incorrect number of files

if 'file1_data' in st.session_state and st.session_state.file1_data:
    if st.button("Compare Documents", use_container_width=True, type="primary"):
        with st.spinner("Analyzing document changes..."):
            text1 = read_pdf_text(st.session_state.file1_data)
            text2 = read_pdf_text(st.session_state.file2_data)

            if text1 is not None and text2 is not None:
                st.session_state['original_text'] = text1
                st.session_state['revised_text'] = text2
                diff = generate_diff_html(text1, text2, st.session_state.file1_data.name, st.session_state.file2_data.name)
                st.session_state['diff_html'] = diff
                st.session_state['summary'] = None # Reset summary on new comparison
            else:
                st.error("Failed to process one or both PDF files.")

if 'diff_html' in st.session_state and st.session_state.diff_html:
    with st.expander("Show/Hide Side-by-Side Diff", expanded=True):
        st.components.v1.html(st.session_state.diff_html, height=600, scrolling=True)

    st.markdown("---")
    st.subheader("🤖 AI-Powered Summary")
    st.markdown("Click the button below for a summary of the key changes.")

    if st.button("✨ Generate Summary", use_container_width=True, disabled=not ai_enabled):
        with st.spinner("Asking Gemini to analyze the changes..."):
            summary = get_ai_summary(st.session_state.original_text, st.session_state.revised_text)
            st.session_state['summary'] = summary

if 'summary' in st.session_state and st.session_state.summary:
    st.markdown("### Summary of Changes:")
    st.markdown(st.session_state.summary)