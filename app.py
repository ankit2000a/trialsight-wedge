import streamlit as st
import fitz  # PyMuPDF
import difflib
import google.generativeai as genai
import os
import time
import re
import itertools # Needed for SequenceMatcher grouping

# --- Page Configuration ---
st.set_page_config(
    page_title="TrialSight: Document Comparator",
    page_icon="🔎",
    layout="wide"
)

# --- App Styling (CSS) ---
# (CSS Styles - collapsed)
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E2E8F0; }
    .stButton>button { border-radius: 0.5rem; }
    .stFileUploader { border: 2px dashed #4A5568; background-color: #1A202C; border-radius: 0.5rem; padding: 2rem; text-align: center; }
    .file-card { background-color: #2D3748; border-radius: 0.5rem; padding: 1rem; border: 1px solid #4A5568; display: flex; justify-content: space-between; align-items: center; }
    /* Diff Table Styles - Adjusted for single line potentially */
    .diff_add { background-color: rgba(16, 185, 129, 0.15); color: #A7F3D0; }
    .diff_chg { background-color: rgba(209, 163, 23, 0.15); color: #FDE68A; }
    .diff_sub { background-color: rgba(239, 68, 68, 0.15); color: #FECACA; text-decoration: none; border-bottom: 1px dotted rgba(239, 68, 68, 0.5); }
    table.diff { font-family: Consolas, 'Courier New', monospace; border-collapse: collapse; width: 100%; font-size: 0.875em; table-layout: fixed; } /* Fixed layout */
    .diff_header { background-color: #374151; color: #E5E7EB; padding: 0.2em 0.5em; font-weight: bold; position: sticky; top: 0; z-index: 10;}
    td { padding: 0.1em 0.4em; vertical-align: top; white-space: pre-wrap; word-wrap: break-word; } /* Allow breaking long words */
    .diff_next { background-color: #4b5563; }
    /* Summary Styles */
    .summary-add { color: #6EE7B7; }
    .summary-del { color: #FCA5A5; text-decoration: line-through; }
    .summary-header { font-weight: bold; margin-top: 0.8em; margin-bottom: 0.3em; }

    /* Loader */
    .loader-container { display: flex; justify-content: center; align-items: center; flex-direction: column; margin: 20px 0; }
    .loader { border: 4px solid #4b5563; border-left-color: #3b82f6; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
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
# (Same as previous version - checks secrets then env var)
ai_enabled = False
api_key = None
try:
    api_key_secret = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets else None
    api_key_env = os.environ.get("GOOGLE_API_KEY")
    api_key = api_key_secret or api_key_env

    if api_key:
        genai.configure(api_key=api_key)
        if 'gemini_model' not in st.session_state or st.session_state.gemini_model is None:
             st.session_state.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
        model = st.session_state.gemini_model
        ai_enabled = True
    else:
        st.warning("Google API Key not found. AI Summary disabled.")
        ai_enabled = False
except Exception as e:
    ai_enabled = False
    st.warning(f"Could not initialize Google AI: {e}")
    if 'gemini_model' in st.session_state: st.session_state.gemini_model = None

# --- Helper Functions ---

@st.cache_data
def extract_and_clean_text(file_bytes, filename="file"):
    """Extracts text and performs HYPER-AGGRESSIVE cleaning, collapsing all whitespace."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = " ".join(page.get_text("text", sort=True) for page in doc) # Join pages with space

        # --- Hyper-Aggressive Cleaning ---
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl') # Fix ligatures
        text = text.lower() # Convert to lowercase
        # Remove line breaks and tabs, replace with space
        text = re.sub(r'[\n\r\t]+', ' ', text)
        # Collapse multiple spaces to a single space
        text = re.sub(r'[ ]+', ' ', text)
        # Optional: Remove punctuation if needed
        # text = re.sub(r'[^\w\s]', '', text) # Removes punctuation except whitespace
        text = text.strip() # Final trim

        if not text:
             print(f"Warning: No text extracted from {filename}")
             return f"ERROR: No text found in PDF {filename}"
        return text
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return f"ERROR: Could not read {filename}. Details: {e}"


def generate_diff_html(text1, text2, filename1="Original", filename2="Revised"):
    """Creates side-by-side HTML diff using the aggressively cleaned text."""
    # NOTE: This diff will NOT preserve original line breaks due to cleaning
    if text1 is None or text2 is None or text1.startswith("ERROR:") or text2.startswith("ERROR:"):
         return "Error: Cannot generate diff (text extraction failed)."
    try:
        # Since text is one long line, splitlines() might just give one item
        # Difflib might still show differences based on internal word/char comparison
        html = difflib.HtmlDiff(wrapcolumn=80, tabsize=4).make_table(
            # Pass as single-element lists if they don't contain newlines from cleaning
            text1.splitlines() if '\n' in text1 else [text1],
            text2.splitlines() if '\n' in text2 else [text2],
            fromdesc=filename1 + " (Cleaned)", # Indicate text is cleaned
            todesc=filename2 + " (Cleaned)"
        )
        style = f"<style>{difflib.HtmlDiff._styles}</style>"
        custom_style = """
        <style>
        table.diff { font-family: Consolas, 'Courier New', monospace; border-collapse: collapse; width: 100%; font-size: 0.875em; table-layout: fixed; }
        .diff_header { background-color: #374151; color: #E5E7EB; padding: 0.2em 0.5em; font-weight: bold; position: sticky; top: 0; z-index: 10;}
        td { padding: 0.1em 0.4em; vertical-align: top; white-space: pre-wrap; word-wrap: break-word; } /* Force wrap */
        .diff_next { background-color: #4b5563; }
        .diff_add { background-color: rgba(16, 185, 129, 0.1); }
        .diff_chg { background-color: rgba(209, 163, 23, 0.1); }
        .diff_sub { background-color: rgba(239, 68, 68, 0.1); text-decoration: none; border-bottom: 1px dotted rgba(239, 68, 68, 0.5); }
        </style>
        """
        # Ensure the container allows scrolling if content overflows
        return f"<div style='overflow-x: auto;'>{style + custom_style + html}</div>"
    except Exception as e:
        print(f"Error generating HTML diff: {e}")
        return f"Error: Failed to generate visual comparison. {e}"


def get_ai_summary(text1_clean, text2_clean):
    """Generates a categorized summary using AI based on hyper-cleaned diff."""
    # --- Input Validation ---
    if not ai_enabled: return "AI Summary feature is not available (API key issue)."
    if text1_clean is None or text2_clean is None or \
       (isinstance(text1_clean, str) and text1_clean.startswith("ERROR:")) or \
       (isinstance(text2_clean, str) and text2_clean.startswith("ERROR:")):
        return "AI Summary cannot be generated: text extraction failed."

    # --- Diff Generation (using hyper-cleaned text) ---
    # Treat the cleaned text as potentially single lines for difflib
    lines1 = text1_clean.splitlines(keepends=True) if '\n' in text1_clean else [text1_clean + '\n']
    lines2 = text2_clean.splitlines(keepends=True) if '\n' in text2_clean else [text2_clean + '\n']
    # Use context (n=3) to help AI understand location better
    diff = list(difflib.unified_diff(lines1, lines2, fromfile='Original_Cleaned', tofile='Revised_Cleaned', n=3))

    # Get ONLY the lines starting with actual content changes (+ or -)
    meaningful_diff_lines = [line for line in diff if line.startswith(('+', '-')) and not line.startswith(('---', '+++')) and line[1:].strip()]

    # --- Handle No Differences Case ---
    if not meaningful_diff_lines:
        # Check if the raw diff had any lines at all (even just headers)
        if diff: # Diff itself wasn't empty
             return "INFO: No substantive content changes found after aggressive cleaning (ignoring formatting, case, and whitespace)."
        else: # Should not happen if texts were identical, but as a fallback
            return "INFO: No textual differences were found between the documents."

    diff_text_for_prompt = "".join(meaningful_diff_lines)

    # --- AI Prompt (v12 - using hyper-cleaned diff) ---
    # Reverting to the simpler 3-category prompt, expecting cleaner input
    prompt = f"""
    Analyze the provided ADDED (+) and DELETED (-) lines/segments, which represent meaningful content changes between two clinical trial protocol versions after aggressive cleaning (lowercase, all whitespace collapsed). Categorize these changes.

    **Instructions:**
    1.  **Clinically Significant Changes:** Identify ADDED (+) or DELETED (-) segments clearly related to:
        * Inclusion/Exclusion criteria
        * Dosage / Treatment schedules
        * Procedures / Assessments
        * Safety reporting
        * Objectives / Endpoints
        List these first, including the full text of the change (+/-). Try to infer context.
    2.  **Other Added Content:** List ALL OTHER provided ADDED (+) segments not in the significant category.
    3.  **Other Deleted Content:** List ALL OTHER provided DELETED (-) segments not in the significant category.
    4.  **IGNORE Noise:** You should not see reformatting noise in the input. Focus only on the content provided.
    5.  **Output Format:** Structure your response EXACTLY like this, using markdown H4 (####) for headers:

        #### Clinically Significant Changes (Added/Deleted Content)
        * [List ONLY significant ADDED (+) or DELETED (-) content here. If none found, state "None found."]

        #### Other Added Content
        * [List ALL OTHER non-blank ADDED (+) content from the input here. If none found, state "None found."]

        #### Other Deleted Content
        * [List ALL OTHER non-blank DELETED (-) content from the input here. If none found, state "None found."]

    **Meaningful Added (+) and Deleted (-) Content Segments (Hyper-Cleaned):**
    ---
    {diff_text_for_prompt[:8000]}
    ---

    Begin Summary:
    """


    try:
        safety_settings = [ {"category": c, "threshold": "BLOCK_LOW_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        generation_config = genai.types.GenerationConfig(temperature=0.1)
        if 'gemini_model' not in st.session_state or st.session_state.gemini_model is None:
             return "ERROR: Gemini model not initialized."
        model = st.session_state.gemini_model
        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)

        # Robust response handling
        if not response.candidates:
            block_reason = "Unknown";
            if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'): block_reason = response.prompt_feedback.block_reason
            return f"ERROR: AI response blocked. Reason: {block_reason}."
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else "Unknown"
        if candidate.content and candidate.content.parts and candidate.content.parts[0].text:
             response_text = candidate.content.parts[0].text.strip()
             return response_text
        else:
             return f"ERROR: AI model returned an empty response. Finish Reason: {finish_reason}"
    except Exception as e:
        error_message = f"ERROR: Failed to get AI summary: {e}"
        if "quota" in str(e).lower() or "429" in str(e): error_message += "\n(Quota exceeded?)"
        # Return error string for display
        return error_message

# --- Main App UI ---
st.title("📄 TrialSight: Document Comparator")
st.markdown("Compare two versions of a document and get an AI-powered summary of content changes.") # Updated
st.markdown("---")

# File Uploader & Display Logic
if not st.session_state.get('file1_data') and not st.session_state.get('file2_data'):
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, key="file_uploader")
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

# --- Comparison Logic ---
if st.session_state.get('file1_data') and st.session_state.get('file2_data'):
    # Show Compare button only if needed
    if not st.session_state.get('diff_html') and not st.session_state.get('processing_comparison'):
        if st.button("Compare Documents", type="primary", use_container_width=True):
            st.session_state.processing_comparison = True
            keys_to_clear = ['diff_html', 'summary', 'original_text', 'revised_text']
            for key in keys_to_clear:
                 if key in st.session_state: del st.session_state[key]
            st.rerun()

    # Execute comparison if flagged
    if st.session_state.get('processing_comparison'):
        with st.spinner("Reading, cleaning, and comparing documents..."):
            time.sleep(0.5) # Ensure spinner shows
            file1 = st.session_state.file1_data
            file2 = st.session_state.file2_data
            if file1 and file2:
                # Use the NEW hyper-aggressive cleaning function
                text1_clean = extract_and_clean_text(file1.getvalue(), file1.name)
                text2_clean = extract_and_clean_text(file2.getvalue(), file2.name)

                # Store the cleaned text (this is what's used for AI summary and now also HTML diff)
                st.session_state['original_text'] = text1_clean
                st.session_state['revised_text'] = text2_clean

                # Generate HTML diff using the cleaned text
                st.session_state['diff_html'] = generate_diff_html(text1_clean, text2_clean, file1.name, file2.name) # Handles internal errors
            else:
                st.session_state['diff_html'] = "Error: File data missing."
                st.session_state['summary'] = None
            st.session_state.processing_comparison = False # Reset flag
            # Rerun only if diff generation didn't return an error string
            if isinstance(st.session_state.get('diff_html'), str) and not st.session_state.get('diff_html', "").startswith("Error:"):
                 st.rerun()


# --- Display Results Section ---
if not st.session_state.get('processing_comparison') and st.session_state.get('diff_html') and isinstance(st.session_state.get('diff_html'), str) and not st.session_state.get('diff_html', "").startswith("Error:"):

    # --- DEBUGGER (Shows the hyper-cleaned text now) ---
    with st.expander("Show/Hide Cleaned Text (Used for Comparison)"):
        col1, col2 = st.columns(2)
        original_display = st.session_state.get('original_text') if isinstance(st.session_state.get('original_text'), str) else "Extraction Error or Not Run"
        revised_display = st.session_state.get('revised_text') if isinstance(st.session_state.get('revised_text'), str) else "Extraction Error or Not Run"
        # Use markdown with code block for potentially long single lines
        with col1: st.subheader("Original (Hyper-Cleaned)"); st.code(original_display, language=None)
        with col2: st.subheader("Revised (Hyper-Cleaned)"); st.code(revised_display, language=None)

    # --- Side-by-Side Diff (using hyper-cleaned text) ---
    st.subheader("Visual Comparison (Based on Cleaned Text)")
    st.markdown("*(Note: Line breaks from original PDF are ignored here to focus on content changes)*")
    # Wrap the HTML component in a div that allows scrolling
    st.components.v1.html(st.session_state.diff_html, height=400, scrolling=True)


    st.markdown("---")

    # --- AI Summary ---
    st.subheader("🤖 AI-Powered Summary of Content Changes") # Updated title
    st.markdown("Click for summary based on cleaned text comparison.")
    # Disable button checks remain the same
    button_disabled = not ai_enabled or (isinstance(st.session_state.get('original_text'), str) and st.session_state.get('original_text', "").startswith("ERROR:")) \
                      or (isinstance(st.session_state.get('revised_text'), str) and st.session_state.get('revised_text', "").startswith("ERROR:")) \
                      or st.session_state.get('original_text') is None or st.session_state.get('revised_text') is None

    summary_button_label = "✨ Get Content Changes Summary" # Updated label
    if st.session_state.get('summary') is not None: summary_button_label = "🔄 Regenerate Summary"

    if st.button(summary_button_label, use_container_width=True, disabled=button_disabled, key="gen_summary_btn"):
        # Use the cleaned texts stored in state
        if st.session_state.get('original_text') is not None and st.session_state.get('revised_text') is not None:
            with st.spinner("Analyzing content changes..."):
                summary_result = get_ai_summary(st.session_state.original_text, st.session_state.revised_text)
                st.session_state['summary'] = summary_result
                st.rerun()
        else: st.error("Cannot generate summary: Cleaned text missing.")

    # --- Display Summary (Using Formatted Display Logic - unchanged) ---
    if st.session_state.get('summary') is not None:
         summary_text = st.session_state.summary
         st.markdown("---") # Add separator
         if summary_text.startswith("ERROR:") or "cannot be generated" in summary_text or "not available" in summary_text:
             st.error(summary_text)
         elif summary_text.startswith("INFO:") or "No textual differences" in summary_text or "No substantive" in summary_text:
              st.info(summary_text.replace("INFO: ", ""))
         else:
            try:
                sections = {}
                current_section_key = None
                headers_map = {
                    "#### Clinically Significant Changes (Added/Deleted Content)": "significant", # Matched prompt
                    "#### Other Added Content": "added", # Matched prompt
                    "#### Other Deleted Content": "deleted" # Matched prompt
                }
                header_display = {
                     "significant": "Clinically Significant Changes (Added/Deleted Content)",
                     "added": "Other Added Content",
                     "deleted": "Other Deleted Content"
                }
                for line in summary_text.splitlines():
                    line_strip = line.strip()
                    matched_header = False
                    for md_header, key in headers_map.items():
                        if line_strip == md_header:
                             current_section_key = key
                             sections[current_section_key] = []
                             matched_header = True
                             break
                    if matched_header: continue
                    if current_section_key and line_strip.startswith('* '):
                         sections[current_section_key].append(line_strip[2:].strip())
                    elif current_section_key and line_strip and "none found" in line_strip.lower():
                         sections[current_section_key].append(line_strip)

                st.markdown("### Categorized Summary of Content Changes:") # Updated title
                for key, display_name in header_display.items():
                    st.markdown(f'<p class="summary-header">{display_name}</p>', unsafe_allow_html=True)
                    if key in sections and sections[key]:
                        items = sections[key]
                        if len(items) == 1 and "none found" in items[0].lower():
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*{items[0]}*")
                        else:
                            for item in items:
                                if item.startswith('+'):
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span class='summary-add'>{item}</span>", unsafe_allow_html=True) # Keep +/-
                                elif item.startswith('-'):
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span class='summary-del'>{item}</span>", unsafe_allow_html=True) # Keep +/-
                                else:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{item}") # Fallback
                    else:
                         st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;*None found.*")
            except Exception as e:
                st.error(f"Failed to parse AI summary. Raw output:\nError: {e}")
                st.markdown(f"```markdown\n{summary_text}\n```")

    elif button_disabled and not st.session_state.get('processing_comparison'):
         if not ai_enabled: st.warning("AI Summary disabled: API Key missing/invalid.")
         elif (isinstance(st.session_state.get('original_text'), str) and st.session_state.get('original_text', "").startswith("ERROR:")) or \
              (isinstance(st.session_state.get('revised_text'), str) and st.session_state.get('revised_text', "").startswith("ERROR:")):
             st.warning("AI Summary disabled: Text extraction failed.")
         elif st.session_state.get('original_text') is None or st.session_state.get('revised_text') is None:
             st.warning("AI Summary disabled: Click 'Compare Documents' first.")

# Handle Errors / Loading State
elif st.session_state.get('diff_html') and isinstance(st.session_state.get('diff_html'), str) and st.session_state.get('diff_html', "").startswith("Error:"):
    st.error(st.session_state.diff_html)
elif st.session_state.get('processing_comparison'):
     st.markdown('<div class="loader-container"><div class="loader"></div><p>Processing...</p></div>', unsafe_allow_html=True)

