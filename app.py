import streamlit as st
import fitz  # PyMuPDF
import difflib
import google.generativeai as genai
import os
import time
import re
# --- NEW LIBRARY ---
import diff_match_patch as dmp_module

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
    /* Diff Table Styles */
    .diff_add { background-color: rgba(16, 185, 129, 0.15); color: #A7F3D0; }
    .diff_chg { background-color: rgba(209, 163, 23, 0.15); color: #FDE68A; }
    .diff_sub { background-color: rgba(239, 68, 68, 0.15); color: #FECACA; text-decoration: none; border-bottom: 1px dotted rgba(239, 68, 68, 0.5); }
    table.diff { font-family: Consolas, 'Courier New', monospace; border-collapse: collapse; width: 100%; font-size: 0.875em; }
    .diff_header { background-color: #374151; color: #E5E7EB; padding: 0.2em 0.5em; font-weight: bold; position: sticky; top: 0; z-index: 10;}
    td { padding: 0.1em 0.4em; vertical-align: top; white-space: pre-wrap; }
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
# (Remains the same - checks secrets then env var)
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
def extract_text_from_bytes(file_bytes, filename="file"):
    """Extracts text with minimal cleaning, preserving line structure for visual diff."""
    # Using the simpler cleaning good for visual diff
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text", sort=True) for page in doc)
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[ \t]+', ' ', text)
        lines = text.splitlines()
        cleaned_lines = [line.strip() for line in lines]
        text = "\n".join(cleaned_lines)
        text = re.sub(r'\n(\s*\n)+', '\n\n', text)
        text = text.strip()
        if not text:
             print(f"Warning: No text extracted from {filename}")
             return f"ERROR: No text found in PDF {filename}"
        return text
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return f"ERROR: Could not read {filename}. Details: {e}"


def generate_diff_html(text1, text2, filename1="Original", filename2="Revised"):
    """Creates side-by-side HTML diff using difflib (for visual presentation)."""
    # Still use difflib for the standard visual HTML table
    if text1 is None or text2 is None or text1.startswith("ERROR:") or text2.startswith("ERROR:"):
         return "Error: Cannot generate diff (text extraction failed)."
    try:
        html = difflib.HtmlDiff(wrapcolumn=80, tabsize=4).make_table(
            text1.splitlines(),
            text2.splitlines(),
            fromdesc=filename1,
            todesc=filename2
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
    except Exception as e:
        print(f"Error generating HTML diff: {e}")
        if isinstance(e, RecursionError):
            return "Error: Failed to generate visual comparison due to text complexity. Try smaller sections."
        return f"Error: Failed to generate visual comparison. {e}"


def get_ai_summary(text1, text2):
    """Generates a categorized summary using AI after diff_match_patch comparison."""
    # --- Input Validation ---
    if not ai_enabled: return "AI Summary feature is not available (API key issue)."
    if text1 is None or text2 is None or (isinstance(text1, str) and text1.startswith("ERROR:")) or (isinstance(text2, str) and text2.startswith("ERROR:")):
        return "AI Summary cannot be generated: text extraction failed."

    # --- Use diff_match_patch for AI input ---
    dmp = dmp_module.diff_match_patch()
    # Use the minimally cleaned text (preserving paragraphs) for DMP
    diffs = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemantic(diffs) # Clean up diffs for better readability

    # --- Extract ONLY true additions and deletions for the AI ---
    # Convert DMP diffs format (operation, text) to +/- lines
    meaningful_diff_lines_for_ai = []
    for op, data in diffs:
        # op is -1 (Delete), 1 (Insert), 0 (Equal)
        data_strip = data.strip()
        if not data_strip: continue # Skip whitespace-only changes

        prefix = ""
        if op == dmp.DIFF_INSERT:
            prefix = "+"
        elif op == dmp.DIFF_DELETE:
            prefix = "-"
        else: # op == dmp.DIFF_EQUAL
            continue # Skip equal parts

        # Add prefix line by line if data contains newlines
        lines = data_strip.splitlines()
        for line in lines:
            line_content = line.strip()
            if line_content: # Ensure line is not blank after splitting/stripping
                meaningful_diff_lines_for_ai.append(f"{prefix}{line_content}\n")


    # --- Handle No Differences Case ---
    if not meaningful_diff_lines_for_ai:
        # Check if the original texts were actually different after basic cleaning
        if text1 != text2:
            return "INFO: No significant content changes detected by diff_match_patch. Differences may be minor whitespace or formatting variations missed by initial cleaning."
        else:
            return "INFO: No textual differences were found between the documents."

    diff_text_for_prompt = "".join(meaningful_diff_lines_for_ai)

    # --- AI Prompt (v13 - Using clean DMP diff) ---
    # Keep the same prompt structure, just updating context description
    prompt = f"""
    Analyze the provided ADDED (+) and DELETED (-) content fragments, which represent meaningful content changes between two clinical trial protocol versions identified by the diff-match-patch algorithm (reformatting noise should be minimal). Categorize these changes.

    **Instructions:**
    1.  **Clinically Significant Changes:** Identify ADDED (+) or DELETED (-) fragments clearly related to:
        * Inclusion/Exclusion criteria
        * Dosage / Treatment schedules
        * Procedures / Assessments
        * Safety reporting
        * Objectives / Endpoints
        List these first, inferring context if possible.
    2.  **Other Added Content:** List ALL OTHER provided ADDED (+) fragments not in the significant category.
    3.  **Other Deleted Content:** List ALL OTHER provided DELETED (-) fragments not in the significant category.
    4.  **Output Format:** Structure your response EXACTLY like this, using markdown H4 (####) for headers:

        #### Clinically Significant Changes (Added/Deleted Content)
        * [List ONLY significant ADDED (+) or DELETED (-) content here. If none found, state "None found."]

        #### Other Added Content
        * [List ALL OTHER non-blank ADDED (+) content from the input here. If none found, state "None found."]

        #### Other Deleted Content
        * [List ALL OTHER non-blank DELETED (-) content from the input here. If none found, state "None found."]

    **Meaningful Added (+) and Deleted (-) Content Fragments (from diff_match_patch):**
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

        # Robust response handling (unchanged)
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
st.markdown("Compare two versions of a document and get an AI-powered summary of meaningful changes.")
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
                # Use the extraction function suitable for HTML diff
                text1 = extract_text_from_bytes(file1.getvalue(), file1.name)
                text2 = extract_text_from_bytes(file2.getvalue(), file2.name)
                st.session_state['original_text'] = text1 # Store this version for display/AI
                st.session_state['revised_text'] = text2 # Store this version for display/AI
                st.session_state['diff_html'] = generate_diff_html(text1, text2, file1.name, file2.name) # Use this version for HTML
            else:
                st.session_state['diff_html'] = "Error: File data missing."
                st.session_state['summary'] = None
            st.session_state.processing_comparison = False # Reset flag
            # Rerun only if diff generation didn't return an error string
            if isinstance(st.session_state.get('diff_html'), str) and not st.session_state.get('diff_html', "").startswith("Error:"):
                 st.rerun()


# --- Display Results Section ---
if not st.session_state.get('processing_comparison') and st.session_state.get('diff_html') and isinstance(st.session_state.get('diff_html'), str) and not st.session_state.get('diff_html', "").startswith("Error:"):

    # --- DEBUGGER (Shows minimally cleaned text) ---
    with st.expander("Show/Hide Cleaned Text (Used for Visual Diff & AI Input)"):
        col1, col2 = st.columns(2)
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
    st.markdown("Click for summary using diff-match-patch analysis.") # Updated text
    button_disabled = not ai_enabled or (isinstance(st.session_state.get('original_text'), str) and st.session_state.get('original_text', "").startswith("ERROR:")) \
                      or (isinstance(st.session_state.get('revised_text'), str) and st.session_state.get('revised_text', "").startswith("ERROR:")) \
                      or st.session_state.get('original_text') is None or st.session_state.get('revised_text') is None

    summary_button_label = "✨ Get Filtered Summary (diff-match-patch)" # Updated label
    if st.session_state.get('summary') is not None: summary_button_label = "🔄 Regenerate Summary"

    if st.button(summary_button_label, use_container_width=True, disabled=button_disabled, key="gen_summary_btn"):
        if st.session_state.get('original_text') is not None and st.session_state.get('revised_text') is not None:
            with st.spinner("Analyzing changes using diff-match-patch..."): # Updated spinner
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
                # Match the H4 headers from the prompt
                headers_map = {
                    # Updated based on latest prompt
                    "#### Clinically Significant Changes (Added/Deleted Content)": "significant",
                    "#### Other Added Content": "added",
                    "#### Other Deleted Content": "deleted"
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
                    # Handle list items and 'None found' text
                    if current_section_key and line_strip.startswith('* '):
                         sections[current_section_key].append(line_strip[2:].strip())
                    elif current_section_key and line_strip and "none found" in line_strip.lower():
                         sections[current_section_key].append(line_strip) # Keep 'None found' as is


                st.markdown("### Categorized Summary (diff-match-patch Filtered):") # Updated title
                for key, display_name in header_display.items():
                    st.markdown(f'<p class="summary-header">{display_name}</p>', unsafe_allow_html=True)
                    if key in sections and sections[key]:
                        items = sections[key]
                        # Check if the only item is 'None found'
                        if len(items) == 1 and "none found" in items[0].lower():
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*{items[0]}*") # Display 'None found' nicely
                        else:
                            for item in items:
                                # Apply styling based on +/- prefix
                                if item.startswith('+'):
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span class='summary-add'>{item}</span>", unsafe_allow_html=True)
                                elif item.startswith('-'):
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span class='summary-del'>{item}</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{item}") # Fallback for unexpected format
                    else:
                         st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;*None found.*") # Default if section is entirely missing


            except Exception as e:
                st.error(f"Failed to parse AI summary format. Raw output:\nError: {e}")
                st.markdown(f"```markdown\n{summary_text}\n```") # Fallback


    # Explain disabled button state
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

