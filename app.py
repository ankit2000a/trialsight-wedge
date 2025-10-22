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
    .diff_sub { background-color: rgba(239, 68, 68, 0.15); color: #FECACA; text-decoration: none; border-bottom: 1px dotted rgba(239, 68, 68, 0.5); } /* Underline */
    table.diff { font-family: Consolas, 'Courier New', monospace; border-collapse: collapse; width: 100%; font-size: 0.875em; }
    .diff_header { background-color: #374151; color: #E5E7EB; padding: 0.2em 0.5em; font-weight: bold; position: sticky; top: 0; z-index: 10;}
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
    # Ensure all keys exist
    keys = ['file1_data', 'file2_data', 'diff_html', 'summary', 'original_text', 'revised_text', 'processing_comparison']
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None
    # Ensure processing_comparison is False initially
    if st.session_state.processing_comparison is None:
         st.session_state.processing_comparison = False

init_session_state()

# --- Gemini API Configuration ---
ai_enabled = False
api_key = None
try:
    # Use secrets.toml as the primary method
    api_key_secret = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets else None
    api_key_env = os.environ.get("GOOGLE_API_KEY") # Fallback to environment variable

    api_key = api_key_secret or api_key_env

    if api_key:
        genai.configure(api_key=api_key)
        # Check if model object needs to be created, avoid recreating if possible
        if 'gemini_model' not in st.session_state or st.session_state.gemini_model is None:
             st.session_state.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
        model = st.session_state.gemini_model
        ai_enabled = True
    else:
        st.warning("Google API Key not found in Streamlit secrets or environment variables. AI Summary disabled.")
        ai_enabled = False
except Exception as e:
    ai_enabled = False
    st.warning(f"Could not initialize Google AI: {e}")
    if 'gemini_model' in st.session_state: st.session_state.gemini_model = None # Reset model on error


# --- Helper Functions ---

@st.cache_data # Keep caching text extraction as it's expensive
def extract_text_from_bytes(file_bytes, filename="file"):
    """Extracts text with minimal cleaning, preserving line structure for diff."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text", sort=True) for page in doc)
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[ \t]+', ' ', text) # Normalize spaces within lines
        lines = text.splitlines()
        cleaned_lines = [line.strip() for line in lines] # Strip ends of lines
        text = "\n".join(cleaned_lines)
        text = re.sub(r'\n(\s*\n)+', '\n\n', text) # Collapse multiple blank lines to one
        text = text.strip()
        return text
    except Exception as e:
        # Don't use st.error inside cache_data function, return error indicator
        print(f"Error reading {filename}: {e}") # Log error to console
        return f"ERROR: Could not read {filename}" # Return error string

def generate_diff_html(text1, text2, filename1="Original", filename2="Revised"):
    """Creates side-by-side HTML diff."""
    if text1 is None or text2 is None or text1.startswith("ERROR:") or text2.startswith("ERROR:"):
         return "Error: Cannot generate diff (text extraction failed)."
    html = difflib.HtmlDiff(wrapcolumn=80, tabsize=4).make_table(
        text1.splitlines(), text2.splitlines(), fromdesc=filename1, todesc=filename2
    )
    style = f"<style>{difflib.HtmlDiff._styles}</style>"
    custom_style = """
    <style>
    table.diff { font-family: Consolas, 'Courier New', monospace; border-collapse: collapse; width: 100%; font-size: 0.875em; }
    .diff_header { background-color: #374151; color: #E5E7EB; padding: 0.2em 0.5em; font-weight: bold; position: sticky; top: 0; z-index: 10;}
    td { padding: 0.1em 0.4em; vertical-align: top; white-space: pre-wrap; }
    .diff_next { background-color: #4b5563; }
    .diff_add { background-color: rgba(16, 185, 129, 0.08); }
    .diff_chg { background-color: rgba(209, 163, 23, 0.08); }
    .diff_sub { background-color: rgba(239, 68, 68, 0.08); text-decoration: none; border-bottom: 1px dotted rgba(239, 68, 68, 0.5); }
    </style>
    """
    return style + custom_style + html

def normalize_text_for_noise_check(text):
    """Aggressively cleans text JUST for checking if content is identical despite formatting."""
    if not isinstance(text, str): return "" # Handle None or other types
    text = text.lower()
    text = re.sub(r'\s+', '', text) # Remove ALL whitespace
    return text

def filter_reformatting_noise(diff_lines):
    """Filters difflib output (+/- lines) to remove reformatting noise."""
    filtered = []
    i = 0
    n = len(diff_lines)
    while i < n:
        line = diff_lines[i]
        # Skip non-change lines
        if not line.startswith(('-', '+')):
            i += 1
            continue

        # Find end of current +/- block
        block_end = i
        while block_end + 1 < n and diff_lines[block_end + 1].startswith(('-', '+')):
            block_end += 1

        # Extract content from block
        deleted_block = [l[1:] for l in diff_lines[i : block_end + 1] if l.startswith('-')]
        added_block = [l[1:] for l in diff_lines[i : block_end + 1] if l.startswith('+')]

        # Compare normalized content
        deleted_content = normalize_text_for_noise_check(" ".join(deleted_block)) # Join with space before normalize
        added_content = normalize_text_for_noise_check(" ".join(added_block)) # Join with space

        # If normalized content is DIFFERENT, keep the *original* lines (if they aren't blank)
        if deleted_content != added_content:
            for k in range(i, block_end + 1):
                if diff_lines[k][1:].strip(): # Check if line has actual content
                    filtered.append(diff_lines[k])

        # Move index past the processed block
        i = block_end + 1
    return filtered


def get_ai_summary(text1, text2):
    """Generates a categorized summary using AI after robust noise filtering."""
    # --- Input Validation ---
    if not ai_enabled: return "AI Summary feature is not available (API key issue)."
    # Check if text extraction failed earlier
    if text1 is None or text2 is None or isinstance(text1, str) and text1.startswith("ERROR:") or isinstance(text2, str) and text2.startswith("ERROR:"):
        return "AI Summary cannot be generated: text extraction failed."

    # --- Diff Generation ---
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    # n=0 ensures only diff lines, simplifies noise filtering logic
    diff = list(difflib.unified_diff(lines1, lines2, fromfile='Original', tofile='Revised', n=0))

    # --- Noise Filtering ---
    raw_diff_lines = [line for line in diff if line.startswith(('+', '-')) and not line.startswith(('---', '+++'))]
    filtered_diff_lines = filter_reformatting_noise(raw_diff_lines)

    # --- Handle No Differences Case ---
    if not filtered_diff_lines:
         if any(line[1:].strip() for line in raw_diff_lines):
             return "INFO: No substantive textual differences found. Changes detected relate primarily to text reformatting or minor whitespace variations."
         else:
            return "INFO: No textual differences were found between the documents after cleaning."

    diff_text_for_prompt = "".join(filtered_diff_lines)

    # --- Prepare AI Call ---
    # Prompt remains the same (v7)
    prompt = f"""
    Analyze the provided ADDED (+) and DELETED (-) lines, which represent meaningful content changes between two clinical trial protocol versions (reformatting noise has been pre-filtered). Categorize these changes.

    **Instructions:**
    1.  **Clinically Significant Lines:** Identify ADDED (+) or DELETED (-) lines clearly related to:
        * Inclusion/Exclusion criteria
        * Dosage / Treatment schedules
        * Procedures / Assessments
        * Safety reporting
        * Objectives / Endpoints
        List these first, inferring context if possible.
    2.  **Other Added Lines:** List ALL OTHER provided ADDED (+) lines not in the significant category.
    3.  **Other Deleted Lines:** List ALL OTHER provided DELETED (-) lines not in the significant category.
    4.  **IGNORE:** Do not mention blank lines or simple whitespace changes.
    5.  **Output Format:** Structure your response EXACTLY like this:

        **Clinically Significant Changes (Added/Deleted Lines):**
        * [List ONLY significant ADDED (+) or DELETED (-) lines here. If none found, state "None found."]

        **Other Added Lines:**
        * [List ALL OTHER non-blank ADDED (+) lines from the input here. If none found, state "None found."]

        **Other Deleted Lines:**
        * [List ALL OTHER non-blank DELETED (-) lines from the input here. If none found, state "None found."]

    **Meaningful Added (+) and Deleted (-) Lines (Reformatting Filtered Out):**
    ---
    {diff_text_for_prompt[:8000]}
    ---

    Begin Summary:
    """

    try:
        # Safety settings and generation config remain the same
        safety_settings = [ {"category": c, "threshold": "BLOCK_LOW_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        generation_config = genai.types.GenerationConfig(temperature=0.1)

        # Make the API call
        if 'gemini_model' not in st.session_state or st.session_state.gemini_model is None:
             return "ERROR: Gemini model not initialized." # Safety check
        model = st.session_state.gemini_model
        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)

        # --- Process Response ---
        if not response.candidates:
            block_reason = "Unknown"
            if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
                 block_reason = response.prompt_feedback.block_reason
            return f"ERROR: AI response blocked. Reason: {block_reason}."

        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else "Unknown"

        if candidate.content and candidate.content.parts and candidate.content.parts[0].text:
             response_text = candidate.content.parts[0].text.strip()
             # Optional: Add warning for non-STOP finish reasons
             # if finish_reason not in ['STOP', 'MAX_TOKENS']:
             #     st.warning(f"AI response finished unexpectedly ({finish_reason}). May be incomplete.")
             return response_text
        else:
             # If text is empty, provide finish reason for debugging
             return f"ERROR: AI model returned an empty response. Finish Reason: {finish_reason}"

    except Exception as e:
        # Catch-all for API communication errors, including quota errors
        error_message = f"ERROR: Failed to get AI summary: {e}"
        # Check specific error details if available
        if "quota" in str(e).lower() or "429" in str(e):
             error_message += "\n(Quota exceeded - please wait a minute or check API key limits)"
        st.error(error_message) # Show error in UI immediately
        return "Summary generation failed due to an API error." # Return generic error for state


# --- Main App UI ---
st.title("📄 TrialSight: Document Comparator")
st.markdown("Compare two versions of a document and get an AI-powered summary of meaningful changes.")
st.markdown("---")

# File Uploader & Display Logic
if not st.session_state.get('file1_data') and not st.session_state.get('file2_data'):
    uploaded_files = st.file_uploader(
        "Upload the original and revised PDF files to compare.", type="pdf", accept_multiple_files=True, key="file_uploader")
    if uploaded_files:
        if len(uploaded_files) != 2: st.warning("Please upload exactly two files.")
        else:
            st.session_state.file1_data = uploaded_files[0]
            st.session_state.file2_data = uploaded_files[1]
            # Clear results immediately on new upload
            keys_to_clear = ['diff_html', 'summary', 'original_text', 'revised_text']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            st.session_state.processing_comparison = False # Ensure reset
            st.rerun()
else:
    # Display file names and clear button
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

# --- Comparison Logic ---
# Show Compare button OR run comparison if needed
if st.session_state.get('file1_data') and st.session_state.get('file2_data'):
    # If results don't exist yet and not currently processing, show button
    if not st.session_state.get('diff_html') and not st.session_state.get('processing_comparison'):
        if st.button("Compare Documents", type="primary", use_container_width=True):
            st.session_state.processing_comparison = True # Set flag to start processing
            # Clear results before starting
            keys_to_clear = ['diff_html', 'summary', 'original_text', 'revised_text']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            st.rerun() # Rerun to show spinner

    # Execute comparison if processing flag is set
    if st.session_state.get('processing_comparison'):
        with st.spinner("Reading, cleaning, and comparing documents..."):
            time.sleep(0.5) # Small delay to ensure spinner appears
            file1 = st.session_state.file1_data
            file2 = st.session_state.file2_data
            if file1 and file2:
                # Extract text (error handling inside function)
                text1 = extract_text_from_bytes(file1.getvalue(), file1.name)
                text2 = extract_text_from_bytes(file2.getvalue(), file2.name)

                # Store extracted text regardless of success for debugging
                st.session_state['original_text'] = text1
                st.session_state['revised_text'] = text2

                # Generate diff only if extraction didn't return error strings
                if isinstance(text1, str) and not text1.startswith("ERROR:") and \
                   isinstance(text2, str) and not text2.startswith("ERROR:"):
                    st.session_state['diff_html'] = generate_diff_html(text1, text2, file1.name, file2.name)
                else:
                    st.session_state['diff_html'] = "Error: Text extraction failed, cannot generate comparison."
                    st.session_state['summary'] = None # Clear summary if diff fails

            else:
                 st.session_state['diff_html'] = "Error: File data missing."
                 st.session_state['summary'] = None

            st.session_state.processing_comparison = False # Reset flag after processing
            # Rerun to display results or error message from diff_html
            st.rerun()


# --- Display Results Section ---
# Check if comparison is done and diff_html exists and is not an error message
if not st.session_state.get('processing_comparison') and st.session_state.get('diff_html') and not st.session_state.get('diff_html', "").startswith("Error:"):

    # --- DEBUGGER ---
    with st.expander("Show/Hide Extracted Text (For Debugging)"):
        col1, col2 = st.columns(2)
        # Display text, handle None or Error strings gracefully
        original_display = st.session_state.get('original_text') if isinstance(st.session_state.get('original_text'), str) else "Extraction Error or Not Run"
        revised_display = st.session_state.get('revised_text') if isinstance(st.session_state.get('revised_text'), str) else "Extraction Error or Not Run"
        with col1: st.subheader("Original (Cleaned)"); st.text_area("Original", original_display, height=200, key="dbg_txt1")
        with col2: st.subheader("Revised (Cleaned)"); st.text_area("Revised", revised_display, height=200, key="dbg_txt2")

    # --- Side-by-Side Diff ---
    with st.expander("Show/Hide Side-by-Side Diff", expanded=True):
        st.components.v1.html(st.session_state.diff_html, height=600, scrolling=True)

    st.markdown("---")

    # --- AI Summary ---
    st.subheader("🤖 AI-Powered Summary")
    st.markdown("Click for a categorized summary filtering formatting noise.")

    # Disable button if AI disabled OR if text extraction failed (check original_text state)
    button_disabled = not ai_enabled or isinstance(st.session_state.get('original_text'), str) and st.session_state.get('original_text', "").startswith("ERROR:") \
                      or isinstance(st.session_state.get('revised_text'), str) and st.session_state.get('revised_text', "").startswith("ERROR:") \
                      or st.session_state.get('original_text') is None or st.session_state.get('revised_text') is None


    # --- MODIFICATION: Generate summary only if button clicked AND summary doesn't exist yet ---
    if st.button("✨ Get Filtered Summary", use_container_width=True, disabled=button_disabled, key="gen_summary_btn"):
        # Check if summary already exists to prevent re-calls
        if st.session_state.get('summary') is None:
            if st.session_state.get('original_text') is not None and st.session_state.get('revised_text') is not None:
                with st.spinner("Analyzing changes (filtering noise)..."):
                    # Call AI summary function
                    summary_result = get_ai_summary(st.session_state.original_text, st.session_state.revised_text)
                    st.session_state['summary'] = summary_result # Store result (could be error string)
                    st.rerun() # Rerun to display the generated summary or error
            else:
                 # This case should ideally be caught by button_disabled, but double-check
                 st.error("Cannot generate summary: Cleaned text missing.")
        # else: # If summary already exists, do nothing on button click to avoid re-call
             # st.info("Summary already generated for this comparison.") # Optional message


    # Display Summary (if it exists)
    if st.session_state.get('summary') is not None: # Check if summary exists (could be None, INFO, ERROR, or actual summary)
         st.markdown("### Categorized Summary (Noise Filtered):")
         summary_text = st.session_state.summary
         # Use specific prefixes/keywords to determine message type
         if summary_text.startswith("ERROR:") or "cannot be generated" in summary_text or "not available" in summary_text:
             st.error(summary_text)
         elif summary_text.startswith("INFO:") or "No textual differences" in summary_text or "No substantive" in summary_text:
              # Remove INFO prefix if present for cleaner display
              display_info = summary_text.replace("INFO: ", "")
              st.info(display_info)
         else:
             # Display the actual summary in a code block
             st.markdown(f"```markdown\n{summary_text}\n```")

    # Explain disabled button state if summary hasn't been generated yet
    elif button_disabled and not st.session_state.get('processing_comparison'):
         if not ai_enabled: st.warning("AI Summary disabled: API Key missing or invalid.")
         elif isinstance(st.session_state.get('original_text'), str) and st.session_state.get('original_text', "").startswith("ERROR:") or \
              isinstance(st.session_state.get('revised_text'), str) and st.session_state.get('revised_text', "").startswith("ERROR:"):
             st.warning("AI Summary disabled: Text extraction failed.")
         elif st.session_state.get('original_text') is None or st.session_state.get('revised_text') is None:
             # This might occur if Compare Documents hasn't been run yet
             st.warning("AI Summary disabled: Please click 'Compare Documents' first.")


# Handle Errors / Loading State
elif st.session_state.get('diff_html') and isinstance(st.session_state.get('diff_html'), str) and st.session_state.get('diff_html', "").startswith("Error:"):
    # Display error from generate_diff_html if it occurred
    st.error(st.session_state.diff_html)
elif st.session_state.get('processing_comparison'):
     # Show spinner if the processing flag is set
     st.markdown('<div class="loader-container"><div class="loader"></div><p>Processing...</p></div>', unsafe_allow_html=True)

