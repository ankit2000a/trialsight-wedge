import streamlit as st
import fitz  # PyMuPDF
import difflib
import google.generativeai as genai
import os
import time
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="TrialSight: Document Comparator",
    page_icon="🔎",
    layout="wide"
)

# --- App Styling (CSS) ---
# (CSS Styles remain the same - collapsed for brevity)
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E2E8F0; }
    .stButton>button { border-radius: 0.5rem; }
    .stFileUploader { border: 2px dashed #4A5568; background-color: #1A202C; border-radius: 0.5rem; padding: 2rem; text-align: center; }
    .file-card { background-color: #2D3748; border-radius: 0.5rem; padding: 1rem; border: 1px solid #4A5568; display: flex; justify-content: space-between; align-items: center; }
    .diff_add { background-color: rgba(16, 185, 129, 0.15); color: #A7F3D0; }
    .diff_chg { background-color: rgba(209, 163, 23, 0.15); color: #FDE68A; }
    .diff_sub { background-color: rgba(239, 68, 68, 0.15); color: #FECACA; text-decoration: line-through; }
    table.diff { font-family: Consolas, 'Courier New', monospace; border-collapse: collapse; width: 100%; font-size: 0.875em; }
    .diff_header { background-color: #374151; color: #E5E7EB; padding: 0.2em 0.5em; font-weight: bold; }
    td { padding: 0.1em 0.4em; vertical-align: top; white-space: pre-wrap; }
    .diff_next { background-color: #4b5563; }
    .loader-container { display: flex; justify-content: center; align-items: center; flex-direction: column; margin: 20px 0; }
    .loader { border: 4px solid #4b5563; border-left-color: #3b82f6; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .summary-box { background-color: #1A202C; border: 1px solid #4A5568; border-radius: 0.5rem; padding: 1.5rem; margin-top: 1rem; }
    .stMarkdown code { white-space: pre-wrap !important; background-color: #1f2937; padding: 0.5em; border-radius: 0.3em; font-size: 0.9em; display: block; }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
def init_session_state():
    for key in ['file1_data', 'file2_data', 'diff_html', 'summary', 'original_text', 'revised_text']:
        if key not in st.session_state: st.session_state[key] = None
    if 'processing_comparison' not in st.session_state: st.session_state.processing_comparison = False
init_session_state()

# --- Gemini API Configuration ---
ai_enabled = False
api_key = None
try:
    api_key_secret = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') else None
    api_key_env = os.environ.get("GOOGLE_API_KEY")
    api_key = api_key_secret or api_key_env # Prioritize secrets

    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        ai_enabled = True
    else:
        st.warning("Google API Key not found. AI Summary disabled.")
        ai_enabled = False
except Exception as e:
    ai_enabled = False
    st.warning(f"Could not initialize Google AI: {e}")

# --- Helper Functions ---

@st.cache_data
def extract_text_from_bytes(file_bytes, filename="file"):
    """Extracts text with minimal cleaning, preserving line structure."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text", sort=True) for page in doc)
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[ \t]+', ' ', text)
        lines = text.splitlines()
        cleaned_lines = [line.strip() for line in lines]
        text = "\n".join(cleaned_lines)
        text = re.sub(r'\n\s*\n+', '\n\n', text) # Collapse multiple blank lines
        text = text.strip()
        return text
    except Exception as e:
        st.error(f"Error reading {filename}: {e}")
        return None

def generate_diff_html(text1, text2, filename1="Original", filename2="Revised"):
    """Creates side-by-side HTML diff using the minimally cleaned text."""
    if text1 is None or text2 is None: return "Error: Cannot generate diff (text extraction failed)."
    d = difflib.HtmlDiff(wrapcolumn=80, tabsize=4)
    html = d.make_table(text1.splitlines(), text2.splitlines(), fromdesc=filename1, todesc=filename2)
    style = f"<style>{difflib.HtmlDiff._styles}</style>"
    custom_style = """
    <style>
    table.diff { font-family: Consolas, 'Courier New', monospace; border-collapse: collapse; width: 100%; font-size: 0.875em; }
    .diff_header { background-color: #374151; color: #E5E7EB; padding: 0.2em 0.5em; font-weight: bold; }
    td { padding: 0.1em 0.4em; vertical-align: top; white-space: pre-wrap; }
    .diff_next { background-color: #4b5563; }
    .diff_add { background-color: rgba(16, 185, 129, 0.1); }
    .diff_chg { background-color: rgba(209, 163, 23, 0.1); }
    .diff_sub { background-color: rgba(239, 68, 68, 0.1); text-decoration: line-through; }
    </style>
    """
    return style + custom_style + html

def normalize_text_for_comparison(text):
    """Aggressively cleans text for content comparison, ignoring formatting."""
    if text is None: return ""
    text = text.lower()
    text = re.sub(r'\s+', '', text) # Remove ALL whitespace
    # Optionally remove punctuation if it causes issues
    # text = re.sub(r'[^\w]', '', text) # Keep only word characters
    return text

def get_ai_summary(text1, text2):
    """Generates a categorized summary focusing on TRUE additions/deletions."""
    if not ai_enabled: return "AI Summary feature is not available."
    if text1 is None or text2 is None: return "AI Summary cannot be generated: text extraction failed."

    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines1, lines2, fromfile='Original', tofile='Revised', n=0))

    raw_diff_lines = [line for line in diff if line.startswith(('+', '-')) and not line.startswith(('---', '+++'))]

    # --- NOISE FILTERING LOGIC ---
    filtered_diff_lines = []
    i = 0
    while i < len(raw_diff_lines):
        line = raw_diff_lines[i]
        if line.startswith('-'):
            # Look ahead for potential corresponding '+' lines
            j = i + 1
            added_lines_content = []
            while j < len(raw_diff_lines) and raw_diff_lines[j].startswith('+'):
                added_lines_content.append(raw_diff_lines[j][1:])
                j += 1

            # Compare normalized content
            deleted_content = normalize_text_for_comparison(line[1:])
            added_content_combined = normalize_text_for_comparison("".join(added_lines_content))

            # If content is the same (just reformatting), skip this block
            if deleted_content == added_content_combined:
                i = j # Skip the matched '+' lines
                continue # Move to the next line after the block

        # If it's a '-' not matched or a standalone '+', keep it (if not blank)
        if line[1:].strip():
            filtered_diff_lines.append(line)
        i += 1
    # --- END NOISE FILTERING ---


    if not filtered_diff_lines:
         if any(line[1:].strip() for line in raw_diff_lines): # Raw diff had non-blank lines, but filter removed them
             return "No substantive textual differences found. Changes detected relate primarily to text reformatting or minor whitespace variations."
         else: # Raw diff was also empty or only whitespace
            return "No textual differences were found between the documents after cleaning."

    diff_text_for_prompt = "".join(filtered_diff_lines)

    # --- PROMPT (using the filtered diff) ---
    prompt = f"""
    Analyze the *meaningful* ADDED (+) and DELETED (-) lines from a comparison between two clinical trial protocol versions (reformatting changes have been excluded). Categorize these line changes into three groups.

    **Instructions:**
    1.  **Clinically Significant Lines:** Identify ADDED (+) or DELETED (-) lines clearly related to:
        * Inclusion/Exclusion criteria
        * Dosage / Treatment schedules
        * Procedures / Assessments
        * Safety reporting
        * Objectives / Endpoints
        List these first, mentioning context/section if possible.
    2.  **Other Added Lines:** List ALL OTHER provided ADDED (+) lines that are not blank and not in the significant category.
    3.  **Other Deleted Lines:** List ALL OTHER provided DELETED (-) lines that are not blank and not in the significant category.
    4.  **IGNORE:** Do NOT report blank lines or lines containing only whitespace (they are excluded from input). Do not analyze changes *within* lines.
    5.  **Output Format:** Structure your response EXACTLY like this:

        **Clinically Significant Changes (Added/Deleted Lines):**
        * [List ONLY significant ADDED (+) or DELETED (-) lines here. If none found, state "None found."]

        **Other Added Lines:**
        * [List ALL OTHER non-blank ADDED (+) lines from the input here. If none found, state "None found."]

        **Other Deleted Lines:**
        * [List ALL OTHER non-blank DELETED (-) lines from the input here. If none found, state "None found."]

    **Detected Meaningful Added (+) and Deleted (-) Non-Blank Lines (Reformatting Excluded):**
    ---
    {diff_text_for_prompt[:8000]}
    ---

    Begin Summary:
    """

    try:
        safety_settings = [ {"category": c, "threshold": "BLOCK_LOW_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        generation_config = genai.types.GenerationConfig(temperature=0.1)
        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)

        # Robust response handling (same as before)
        if not response.candidates:
            block_reason = response.prompt_feedback.block_reason if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason') else "Unknown"
            return f"Error: AI response blocked. Reason: {block_reason}."
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else "Unknown"
        if candidate.content and candidate.content.parts and candidate.content.parts[0].text:
             response_text = candidate.content.parts[0].text.strip()
             # if finish_reason not in ['STOP', 'MAX_TOKENS']: st.warning(f"AI response finished unexpectedly ({finish_reason}).") # Optional warning
             return response_text
        else:
             return f"Error: AI model returned an empty response. Finish Reason: {finish_reason}"
    except Exception as e:
        error_message = f"Error communicating with the AI model: {e}"
        if "quota" in str(e).lower() or "429" in str(e):
             error_message += "\nQuota issue likely. Check API key/limits or wait."
        return error_message

# --- Main App UI ---
st.title("📄 TrialSight: Document Comparator")
st.markdown("Compare two versions of a document to see changes and get an AI-powered summary.")
st.markdown("---")

# File Uploader & Display Logic (same as before)
if not st.session_state.get('file1_data') and not st.session_state.get('file2_data'):
    uploaded_files = st.file_uploader(
        "Upload the original and revised PDF files to compare.", type="pdf", accept_multiple_files=True, key="file_uploader")
    if uploaded_files:
        if len(uploaded_files) != 2: st.warning("Please upload exactly two files.")
        else:
            st.session_state.file1_data = uploaded_files[0]
            st.session_state.file2_data = uploaded_files[1]
            keys_to_clear = ['diff_html', 'summary', 'original_text', 'revised_text']
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
        keys_to_clear = ['file1_data', 'file2_data', 'diff_html', 'summary', 'original_text', 'revised_text', 'processing_comparison']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

# --- Comparison Logic --- (same as before)
if st.session_state.get('file1_data') and st.session_state.get('file2_data'):
    if not st.session_state.get('diff_html') and not st.session_state.get('processing_comparison'):
        if st.button("Compare Documents", type="primary", use_container_width=True):
            st.session_state.processing_comparison = True
            keys_to_clear = ['diff_html', 'summary', 'original_text', 'revised_text']
            for key in keys_to_clear:
                 if key in st.session_state: del st.session_state[key]
            st.rerun()

    if st.session_state.get('processing_comparison'):
        with st.spinner("Reading, cleaning, and comparing documents..."):
            file1 = st.session_state.file1_data
            file2 = st.session_state.file2_data
            if file1 and file2:
                text1 = extract_text_from_bytes(file1.getvalue(), file1.name)
                text2 = extract_text_from_bytes(file2.getvalue(), file2.name)
                if text1 is not None and text2 is not None:
                    st.session_state['original_text'] = text1
                    st.session_state['revised_text'] = text2
                    st.session_state['diff_html'] = generate_diff_html(text1, text2, file1.name, file2.name)
                else:
                    st.session_state['diff_html'] = None; st.session_state['summary'] = None
            else: st.error("File data missing."); st.session_state['diff_html'] = None; st.session_state['summary'] = None
            st.session_state.processing_comparison = False
            if st.session_state.get('diff_html') and "Error:" not in st.session_state.diff_html:
                 st.rerun()


# --- Display Results Section --- (same as before)
if not st.session_state.get('processing_comparison') and st.session_state.get('diff_html') and "Error:" not in st.session_state.get('diff_html', ""):
    # --- DEBUGGER ---
    with st.expander("Show/Hide Extracted Text (For Debugging)"):
        col1, col2 = st.columns(2)
        with col1: st.subheader("Original (Cleaned)"); st.text_area("Original", st.session_state.get('original_text', ''), height=200, key="dbg_txt1")
        with col2: st.subheader("Revised (Cleaned)"); st.text_area("Revised", st.session_state.get('revised_text', ''), height=200, key="dbg_txt2")

    # --- Side-by-Side Diff ---
    with st.expander("Show/Hide Side-by-Side Diff", expanded=True):
        st.components.v1.html(st.session_state.diff_html, height=600, scrolling=True)

    st.markdown("---")

    # --- AI Summary ---
    st.subheader("🤖 AI-Powered Summary")
    st.markdown("Click for a categorized summary of added/deleted lines.")
    button_disabled = not ai_enabled or st.session_state.get('original_text') is None or st.session_state.get('revised_text') is None
    if st.button("✨ Get Categorized Line Summary", use_container_width=True, disabled=button_disabled, key="gen_summary_btn"):
        if st.session_state.get('original_text') is not None and st.session_state.get('revised_text') is not None:
            with st.spinner("Analyzing changes..."):
                summary = get_ai_summary(st.session_state.original_text, st.session_state.revised_text)
                st.session_state['summary'] = summary
                st.rerun()
        else: st.error("Cannot generate summary: Cleaned text missing.")

    # Display Summary
    if st.session_state.get('summary'):
         st.markdown("### Categorized Summary of Added/Deleted Lines:")
         summary_text = st.session_state.summary
         if summary_text.startswith("Error:") or "cannot be generated" in summary_text or "not available" in summary_text:
             st.error(summary_text)
         elif "No textual differences" in summary_text or "No substantive" in summary_text:
              st.info(summary_text)
         else:
             st.markdown(f"```markdown\n{summary_text}\n```")

    elif button_disabled and not st.session_state.get('processing_comparison'):
         if not ai_enabled: st.warning("AI Summary disabled: API Key missing.")
         elif st.session_state.get('original_text') is None or st.session_state.get('revised_text') is None:
             st.warning("AI Summary disabled: Text extraction failed.")

# Handle Errors / Loading State (same as before)
elif st.session_state.get('diff_html') and "Error:" in st.session_state.get('diff_html', ""):
    st.error(st.session_state.diff_html)
elif st.session_state.get('processing_comparison'):
     st.markdown('<div class="loader-container"><div class="loader"></div><p>Processing...</p></div>', unsafe_allow_html=True)

