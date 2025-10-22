import streamlit as st
import fitz  # PyMuPDF
import difflib
import google.generativeai as genai
import os
import time
import re  # Still need re for basic cleaning

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
    /* Updated Diff Styles for Clarity */
    .diff_add { background-color: rgba(16, 185, 129, 0.15); color: #A7F3D0; text-decoration: none; } /* Lighter Green */
    .diff_chg { background-color: rgba(209, 163, 23, 0.15); color: #FDE68A; } /* Lighter Yellow */
    .diff_sub { background-color: rgba(239, 68, 68, 0.15); color: #FECACA; text-decoration: line-through; } /* Lighter Red */
    table.diff { font-family: monospace; border-collapse: collapse; width: 100%; font-size: 0.85em; } /* Slightly smaller font */
    .diff_header { background-color: #374151; color: #E5E7EB; padding: 0.2em 0.5em; }
    td { padding: 0.1em 0.3em; vertical-align: top; }

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
    /* Style for markdown code blocks used in summary */
     .stMarkdown code {
        white-space: pre-wrap !important; /* Allow wrapping in code blocks */
        background-color: #1f2937; /* Darker background for code */
        padding: 0.5em;
        border-radius: 0.3em;
        font-size: 0.9em;
        display: block; /* Make it block for better layout */
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
def init_session_state():
    # Keep it simple for now
    for key in ['file1_data', 'file2_data', 'diff_html', 'summary', 'original_text', 'revised_text']:
        if key not in st.session_state:
            st.session_state[key] = None
    if 'processing_comparison' not in st.session_state: # Use specific flag
        st.session_state.processing_comparison = False

init_session_state()

# --- Gemini API Configuration ---
ai_enabled = False
api_key = None # Initialize api_key
try:
    # Attempt to get API key from environment variables first, then secrets
    api_key_env = os.environ.get("GOOGLE_API_KEY")
    api_key_secret = None

    if hasattr(st, 'secrets'): # Check if st.secrets exists
         secrets = st.secrets
         if "GOOGLE_API_KEY" in secrets:
            api_key_secret = secrets["GOOGLE_API_KEY"]

    # Prioritize secrets file if it exists, otherwise use environment variable
    if api_key_secret:
        api_key = api_key_secret
        # source = "Streamlit secrets" # Debug info removed
    elif api_key_env:
        api_key = api_key_env
        # source = "Environment variable" # Debug info removed
    # else: # Debug info removed
        # source = "None found" # Debug info removed


    if api_key:
        # --- DEBUG LINES REMOVED ---
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        ai_enabled = True
    else:
        st.warning("Google API Key not found in environment variables or Streamlit secrets. The AI Summary feature is disabled.")
        ai_enabled = False

except ImportError:
    st.warning("Streamlit secrets management not available. Ensure API key is set as an environment variable.")
    ai_enabled = False
except AttributeError:
    st.warning("Streamlit secrets attribute not found. Ensure API key is set as an environment variable.")
    ai_enabled = False
except Exception as e:
    ai_enabled = False
    st.warning(f"Could not initialize Google AI: {e}")


# --- Helper Functions ---

@st.cache_data
def extract_text_from_bytes(file_bytes, filename="file"):
    """Extracts and performs MINIMAL cleaning text from PDF bytes."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        # Extract text page by page, trying to preserve structure with newlines
        # sort=True can help maintain reading order on complex layouts
        text = "\n".join(page.get_text("text", sort=True) for page in doc)

        # --- Minimal Cleaning ---
        # Fix common PDF ligature issues
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')

        # Basic whitespace cleanup: replace multiple spaces/tabs with single space within lines
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove leading/trailing whitespace from each line BUT keep blank lines if they were there
        lines = text.splitlines()
        cleaned_lines = [line.strip() for line in lines]
        text = "\n".join(cleaned_lines)
        # Collapse multiple consecutive blank lines into a single blank line
        text = re.sub(r'\n\s*\n+', '\n\n', text)

        # Remove leading/trailing whitespace from the whole text block
        text = text.strip()

        return text
    except Exception as e:
        st.error(f"Error reading {filename}: {e}")
        return None


def generate_diff_html(text1, text2, filename1="Original", filename2="Revised"):
    """Creates a side-by-side HTML diff of two texts."""
    if text1 is None or text2 is None:
        return "Error: Cannot generate diff because text extraction failed."

    d = difflib.HtmlDiff(wrapcolumn=80, tabsize=4) # Added tabsize
    html = d.make_table(text1.splitlines(), text2.splitlines(), fromdesc=filename1, todesc=filename2)
    # Get the style from the HtmlDiff class itself
    style = f"<style>{difflib.HtmlDiff._styles}</style>" # Use built-in styles
    # Add custom overrides if needed (adjusting colors slightly)
    custom_style = """
    <style>
    table.diff { font-family: Consolas, 'Courier New', monospace; border-collapse: collapse; width: 100%; font-size: 0.875em; } /* Slightly larger */
    .diff_header { background-color: #374151; color: #E5E7EB; padding: 0.2em 0.5em; font-weight: bold; }
    td { padding: 0.1em 0.4em; vertical-align: top; white-space: pre-wrap; } /* Ensure wrapping */
    .diff_next { background-color: #4b5563; } /* Context control */
    .diff_add { background-color: rgba(16, 185, 129, 0.1); } /* Lighter Green */
    .diff_chg { background-color: rgba(209, 163, 23, 0.1); } /* Lighter Yellow */
    .diff_sub { background-color: rgba(239, 68, 68, 0.1); text-decoration: line-through; } /* Lighter Red */
    </style>
    """
    # Combine default styles with overrides
    return style + custom_style + html


def get_ai_summary(text1, text2):
    """Generates a summary categorizing ADDED/DELETED lines using Gemini."""
    if not ai_enabled:
        return "AI Summary feature is not available."
    if text1 is None or text2 is None:
        return "AI Summary cannot be generated because text extraction failed."

    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    # n=0 means no context lines, only diffs
    diff = list(difflib.unified_diff(lines1, lines2, fromfile='Original', tofile='Revised', n=0))

    # Filter for non-blank added/deleted lines
    diff_lines = [line for line in diff if line.startswith(('+', '-')) and not line.startswith(('---', '+++')) and line[1:].strip()]

    if not diff_lines:
         original_diff_lines_all = [line for line in diff if line.startswith(('+', '-')) and not line.startswith(('---', '+++'))]
         if original_diff_lines_all: # Some diff lines existed but were only whitespace
             return "No substantive textual differences (additions or deletions of non-blank lines) were found. Only whitespace or blank line changes detected."
         else: # No diff lines at all
            return "No textual differences were found between the documents after cleaning."

    diff_text_for_prompt = "".join(diff_lines)

    # --- FINAL PROMPT v3 ---
    prompt = f"""
    Analyze the ADDED (+) and DELETED (-) lines from a comparison between an original and a revised clinical trial protocol. Categorize these line changes based on their likely clinical significance.

    **Instructions:**
    1.  **Clinically Significant Lines:** Identify ADDED (+) or DELETED (-) lines that clearly relate to these key areas:
        * Inclusion/Exclusion criteria
        * Dosage information / Treatment schedules
        * Study procedures / Assessments
        * Safety reporting requirements
        * Key objectives / Endpoints
        Infer the context/section if possible (e.g., "Inclusion Criteria").
    2.  **Other Added Lines:** List ALL OTHER ADDED (+) lines that are not blank and do not fall into the clinically significant category above.
    3.  **Other Deleted Lines:** List ALL OTHER DELETED (-) lines that are not blank and do not fall into the clinically significant category above.
    4.  **IGNORE:** Do NOT report lines that only show changes *within* them (these are not provided in the input). Do NOT report blank lines or lines containing only whitespace.
    5.  **Output Format:** Structure your response EXACTLY like this:

        **Clinically Significant Changes (Added/Deleted Lines):**
        * [List ONLY the significant ADDED (+) or DELETED (-) lines here, mentioning context if possible. If none, state "None found."]

        **Other Added Lines:**
        * [List ALL OTHER non-blank ADDED (+) lines here. If none, state "None found."]

        **Other Deleted Lines:**
        * [List ALL OTHER non-blank DELETED (-) lines here. If none, state "None found."]

    **Detected Added (+) and Deleted (-) Non-Blank Lines:**
    ---
    {diff_text_for_prompt[:8000]}
    ---

    Begin Summary:
    """

    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        generation_config = genai.types.GenerationConfig(
            temperature=0.1, # Lower temp for more deterministic output
            # max_output_tokens=1024 # Limit output size if needed
            )

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        if not response.candidates:
             block_reason = "Unknown"
             safety_ratings_str = "N/A"
             if hasattr(response, 'prompt_feedback'):
                 feedback = response.prompt_feedback
                 if hasattr(feedback, 'block_reason'): block_reason = feedback.block_reason
                 if hasattr(feedback, 'safety_ratings'): safety_ratings_str = ", ".join([f"{rating.category}: {rating.probability}" for rating in feedback.safety_ratings])
             return f"Error: AI response blocked. Reason: {block_reason}. Safety: [{safety_ratings_str}]."

        if response.text:
            return response.text.strip()
        else:
            finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
            if finish_reason == 'SAFETY':
                 safety_ratings_str = ", ".join([f"{rating.category}: {rating.probability}" for rating in response.candidates[0].safety_ratings])
                 return f"Error: AI empty response (Safety). Reason: {finish_reason}. Ratings: [{safety_ratings_str}]"
            else:
                 return f"Error: AI model returned an empty response. Finish Reason: {finish_reason}"

    except Exception as e:
        error_message = f"Error communicating with the AI model: {e}"
        if "quota" in str(e).lower() or "429" in str(e):
             error_message += "\nQuota issue likely. Check API key usage/limits or wait and retry."
        return error_message

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
            # Clear previous results immediately
            keys_to_clear = ['diff_html', 'summary', 'original_text', 'revised_text']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            st.session_state.processing_comparison = False # Reset flag
            st.rerun()
else:
    # Display uploaded file names
    col1, col2 = st.columns(2)
    file1_name = st.session_state.file1_data.name if st.session_state.get('file1_data') else "N/A"
    file2_name = st.session_state.file2_data.name if st.session_state.get('file2_data') else "N/A"
    with col1:
        st.success(f"Original: **{file1_name}**")
    with col2:
        st.success(f"Revised: **{file2_name}**")

    # Clear Files Button
    if st.button("Clear Files and Start Over"):
        keys_to_clear = ['file1_data', 'file2_data', 'diff_html', 'summary',
                         'original_text', 'revised_text', 'processing_comparison']
        for key in keys_to_clear:
             if key in st.session_state: del st.session_state[key]
        st.rerun()

# --- Comparison Logic ---
# Run comparison automatically if files are loaded but results aren't computed yet
# OR if the compare button is explicitly clicked (even if results exist, allows re-compare)
should_compare = False
compare_button_clicked = False

if st.session_state.get('file1_data') and st.session_state.get('file2_data'):
    # Show Compare button only if results don't exist yet
    if not st.session_state.get('diff_html') and not st.session_state.get('processing_comparison', False):
         if st.button("Compare Documents", type="primary", use_container_width=True):
            should_compare = True
            compare_button_clicked = True
            st.session_state.processing_comparison = True # Set flag for spinner
            # Clear results before re-comparing on button click
            st.session_state.diff_html = None
            st.session_state.summary = None
            st.session_state.original_text = None
            st.session_state.revised_text = None
            st.rerun() # Rerun immediately for spinner

# Execute comparison if flagged
if st.session_state.get('processing_comparison', False):
     with st.spinner("Reading, cleaning, and comparing documents..."):
        file1 = st.session_state.file1_data
        file2 = st.session_state.file2_data

        if file1 and file2:
            file1_bytes = file1.getvalue()
            file2_bytes = file2.getvalue()

            text1 = extract_text_from_bytes(file1_bytes, file1.name)
            text2 = extract_text_from_bytes(file2_bytes, file2.name)

            if text1 is not None and text2 is not None:
                st.session_state['original_text'] = text1
                st.session_state['revised_text'] = text2
                st.session_state['diff_html'] = generate_diff_html(text1, text2, file1.name, file2.name)
            else:
                st.error("Text extraction failed. Cannot compare.")
                st.session_state['diff_html'] = None # Clear diff on error
                st.session_state['summary'] = None # Clear summary on error

        else:
             st.error("File data missing, cannot process comparison.")
             st.session_state['diff_html'] = None
             st.session_state['summary'] = None

        st.session_state.processing_comparison = False # Reset flag
        # Rerun only if successful to show results
        if st.session_state.get('diff_html') and "Error:" not in st.session_state.diff_html:
             st.rerun()

# --- Display Results Section ---
# Only display if comparison is done and diff_html is valid
if not st.session_state.get('processing_comparison', False) and st.session_state.get('diff_html') and "Error:" not in st.session_state.get('diff_html', ""):

    # --- DEBUGGER ---
    with st.expander("Show/Hide Extracted Text (For Debugging)"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Text from Original (Cleaned)")
            st.text_area("Original Text", st.session_state.get('original_text', 'N/A'), height=200, key="debug_text1")
        with col2:
            st.subheader("Text from Revised (Cleaned)")
            st.text_area("Revised Text", st.session_state.get('revised_text', 'N/A'), height=200, key="debug_text2")

    # --- Side-by-Side Diff ---
    with st.expander("Show/Hide Side-by-Side Diff", expanded=True):
        st.components.v1.html(st.session_state.diff_html, height=600, scrolling=True)

    st.markdown("---")

    # --- AI Summary ---
    st.subheader("🤖 AI-Powered Summary")
    st.markdown("Click the button below for a categorized summary of added/deleted lines.")

    button_disabled = not ai_enabled or st.session_state.get('original_text') is None or st.session_state.get('revised_text') is None

    if st.button("✨ Get Categorized Line Summary", use_container_width=True, disabled=button_disabled, key="generate_summary"):
        with st.spinner("Analyzing and categorizing added/deleted lines..."):
            summary = get_ai_summary(st.session_state.original_text, st.session_state.revised_text)
            st.session_state['summary'] = summary
            st.rerun()

    if st.session_state.get('summary'):
         st.markdown("### Categorized Summary of Added/Deleted Lines:")
         summary_text = st.session_state.summary
         if "Error:" in summary_text:
             st.error(summary_text)
         else:
             st.markdown(f"```markdown\n{summary_text}\n```") # Display in code block

    elif button_disabled:
         if not ai_enabled:
             st.warning("AI Summary button disabled: API Key not configured.")
         elif st.session_state.get('original_text') is None or st.session_state.get('revised_text') is None:
             st.warning("AI Summary button disabled: Text extraction failed.")

# Handle case where comparison failed (diff_html contains "Error:")
elif st.session_state.get('diff_html') and "Error:" in st.session_state.get('diff_html', ""):
    st.error(st.session_state.diff_html) # Display the diff generation error

# Show spinner if processing flag is somehow still true but diff hasn't rendered
elif st.session_state.get('processing_comparison', False):
     st.markdown('<div class="loader-container"><div class="loader"></div><p>Processing...</p></div>', unsafe_allow_html=True)

