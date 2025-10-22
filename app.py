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
    /* Diff Table Styles */
    .diff_add { background-color: rgba(16, 185, 129, 0.08); }
    .diff_chg { background-color: rgba(209, 163, 23, 0.08); }
    .diff_sub { background-color: rgba(239, 68, 68, 0.08); text-decoration: none; border-bottom: 1px dotted rgba(239, 68, 68, 0.5); }
    table.diff { font-family: Consolas, 'Courier New', monospace; border-collapse: collapse; width: 100%; font-size: 0.875em; }
    .diff_header { background-color: #374151; color: #E5E7EB; padding: 0.2em 0.5em; font-weight: bold; position: sticky; top: 0; z-index: 10;}
    td { padding: 0.1em 0.4em; vertical-align: top; white-space: pre-wrap; }
    .diff_next { background-color: #4b5563; }
    /* Summary Styles */
    .summary-add { color: #6EE7B7; } /* Green for additions */
    .summary-del { color: #FCA5A5; text-decoration: line-through; } /* Red for deletions */
    .summary-header { font-weight: bold; margin-top: 0.8em; margin-bottom: 0.3em; }

    /* Loader */
    .loader-container { display: flex; justify-content: center; align-items: center; flex-direction: column; margin: 20px 0; }
    .loader { border: 4px solid #4b5563; border-left-color: #3b82f6; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .stMarkdown code { /* Keep code block style if AI fails format */
        white-space: pre-wrap !important; background-color: #1f2937; padding: 0.5em; border-radius: 0.3em; font-size: 0.9em; display: block;
    }
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
    """Extracts text, preserving paragraphs (double newlines)."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0])) # Sort blocks
            page_text = ""
            last_y1 = 0
            for b in blocks:
                x0, y0, x1, y1, block_text, block_no, block_type = b
                # Add paragraph break if significant vertical space, ignore if it's the first block on page
                if y0 - last_y1 > 15 and page_text:
                    page_text += "\n\n"
                cleaned_block = block_text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
                cleaned_block = re.sub(r'[ \t]+', ' ', cleaned_block)
                cleaned_block = cleaned_block.strip()
                # Append if block has content
                if cleaned_block:
                    page_text += cleaned_block + " " # Add space between blocks
                last_y1 = y1
            # Add page break marker (optional, could help context)
            # text += page_text.strip() + f"\n\n--- Page {page_num + 1} ---\n\n"
            text += page_text.strip() + "\n\n" # Simpler page break

        # Final cleanup
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text) # Collapse multiple newlines to max two
        text = text.strip()

        if not text:
             print(f"Warning: No text extracted from {filename}")
             return f"ERROR: No text found in PDF {filename}"
        return text
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return f"ERROR: Could not read {filename}. Details: {e}"


def generate_diff_html(text1, text2, filename1="Original", filename2="Revised"):
    """Creates side-by-side HTML diff using paragraph-preserved text."""
    if text1 is None or text2 is None or text1.startswith("ERROR:") or text2.startswith("ERROR:"):
         return "Error: Cannot generate diff (text extraction failed)."
    try:
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()
        html = difflib.HtmlDiff(wrapcolumn=80, tabsize=4).make_table(
            lines1, lines2, fromdesc=filename1, todesc=filename2
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
        return f"Error: Failed to generate visual comparison. {e}"

# Keep the robust noise filter function
def normalize_text_for_noise_check(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\s+', '', text)
    return text

def filter_reformatting_noise(diff_lines):
    filtered = []
    i = 0
    n = len(diff_lines)
    while i < n:
        line = diff_lines[i]
        if not line.startswith(('-', '+')): i += 1; continue
        block_end = i
        while block_end + 1 < n and diff_lines[block_end + 1].startswith(('-', '+')):
            block_end += 1
        deleted_block = [l[1:] for l in diff_lines[i : block_end + 1] if l.startswith('-')]
        added_block = [l[1:] for l in diff_lines[i : block_end + 1] if l.startswith('+')]
        deleted_content = normalize_text_for_noise_check(" ".join(deleted_block))
        added_content = normalize_text_for_noise_check(" ".join(added_block))
        if deleted_content != added_content:
            for k in range(i, block_end + 1):
                if diff_lines[k][1:].strip():
                    filtered.append(diff_lines[k])
        i = block_end + 1
    return filtered


def get_ai_summary(text1, text2):
    """Generates a categorized summary using AI after robust noise filtering."""
    # --- Input Validation ---
    if not ai_enabled: return "AI Summary feature is not available (API key issue)."
    if text1 is None or text2 is None or (isinstance(text1, str) and text1.startswith("ERROR:")) or (isinstance(text2, str) and text2.startswith("ERROR:")):
        return "AI Summary cannot be generated: text extraction failed."

    # --- Diff Generation & Noise Filtering ---
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines1, lines2, fromfile='Original', tofile='Revised', n=0)) # n=0 important for filter
    raw_diff_lines = [line for line in diff if line.startswith(('+', '-')) and not line.startswith(('---', '+++'))]
    filtered_diff_lines = filter_reformatting_noise(raw_diff_lines) # Apply robust filter

    # --- Handle No Differences Case ---
    if not filtered_diff_lines:
         if any(line[1:].strip() for line in raw_diff_lines):
             return "INFO: No substantive textual differences found. Changes detected relate primarily to text reformatting."
         else:
            return "INFO: No textual differences were found between the documents after cleaning."

    diff_text_for_prompt = "".join(filtered_diff_lines)

    # --- Prompt v10 (Keep the 3 categories, simpler instructions) ---
    prompt = f"""
    Analyze the provided ADDED (+) and DELETED (-) lines, which represent meaningful content changes between two clinical trial protocol versions (reformatting noise has been pre-filtered). Categorize these changes into three groups.

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
    4.  **Output Format:** Structure your response EXACTLY like this, using markdown H4 (####) for headers:

        #### Clinically Significant Changes (Added/Deleted Lines)
        * [List ONLY significant ADDED (+) or DELETED (-) lines here. If none found, state "None found."]

        #### Other Added Lines
        * [List ALL OTHER non-blank ADDED (+) lines from the input here. If none found, state "None found."]

        #### Other Deleted Lines
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
             return "ERROR: Gemini model not initialized."
        model = st.session_state.gemini_model
        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)

        # --- Process Response --- (Same robust handling as before)
        if not response.candidates:
            block_reason = "Unknown"
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
        # Return error string for display, don't show Streamlit error directly in summary area
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
    if not st.session_state.get('diff_html') and not st.session_state.get('processing_comparison'):
        if st.button("Compare Documents", type="primary", use_container_width=True):
            st.session_state.processing_comparison = True
            keys_to_clear = ['diff_html', 'summary', 'original_text', 'revised_text']
            for key in keys_to_clear:
                 if key in st.session_state: del st.session_state[key]
            st.rerun()

    if st.session_state.get('processing_comparison'):
        with st.spinner("Reading, cleaning, and comparing documents..."):
            time.sleep(0.5) # Ensure spinner shows
            file1 = st.session_state.file1_data
            file2 = st.session_state.file2_data
            if file1 and file2:
                text1 = extract_text_from_bytes(file1.getvalue(), file1.name)
                text2 = extract_text_from_bytes(file2.getvalue(), file2.name)
                st.session_state['original_text'] = text1
                st.session_state['revised_text'] = text2
                st.session_state['diff_html'] = generate_diff_html(text1, text2, file1.name, file2.name)
            else: st.session_state['diff_html'] = "Error: File data missing."; st.session_state['summary'] = None
            st.session_state.processing_comparison = False
            if isinstance(st.session_state.get('diff_html'), str) and not st.session_state.get('diff_html', "").startswith("Error:"):
                 st.rerun()


# --- Display Results Section ---
if not st.session_state.get('processing_comparison') and st.session_state.get('diff_html') and isinstance(st.session_state.get('diff_html'), str) and not st.session_state.get('diff_html', "").startswith("Error:"):

    # --- DEBUGGER ---
    with st.expander("Show/Hide Extracted Text (Cleaned for Diff)"): # Renamed
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
    st.markdown("Click for summary of significant changes (noise filtered by paragraph).")
    button_disabled = not ai_enabled or (isinstance(st.session_state.get('original_text'), str) and st.session_state.get('original_text', "").startswith("ERROR:")) \
                      or (isinstance(st.session_state.get('revised_text'), str) and st.session_state.get('revised_text', "").startswith("ERROR:")) \
                      or st.session_state.get('original_text') is None or st.session_state.get('revised_text') is None

    summary_button_label = "✨ Get Filtered Summary (Paragraph Aware)"
    if st.session_state.get('summary') is not None: summary_button_label = "🔄 Regenerate Summary"

    if st.button(summary_button_label, use_container_width=True, disabled=button_disabled, key="gen_summary_btn"):
        if st.session_state.get('original_text') is not None and st.session_state.get('revised_text') is not None:
            with st.spinner("Analyzing changes..."):
                summary_result = get_ai_summary(st.session_state.original_text, st.session_state.revised_text)
                st.session_state['summary'] = summary_result
                st.rerun()
        else: st.error("Cannot generate summary: Cleaned text missing.")

    # --- NEW SUMMARY DISPLAY LOGIC ---
    if st.session_state.get('summary') is not None:
         summary_text = st.session_state.summary
         st.markdown("---") # Add a separator

         # Handle Info/Error messages first
         if summary_text.startswith("ERROR:") or "cannot be generated" in summary_text or "not available" in summary_text:
             st.error(summary_text)
         elif summary_text.startswith("INFO:") or "No textual differences" in summary_text or "No substantive" in summary_text:
              st.info(summary_text.replace("INFO: ", ""))
         # Attempt to parse and display formatted summary
         else:
            try:
                sections = {}
                current_section = None
                lines = summary_text.splitlines()

                # Define expected headers based on the prompt
                headers = [
                    "#### Clinically Significant Changes (Added/Deleted Lines)",
                    "#### Other Added Lines",
                    "#### Other Deleted Lines"
                ]

                # Parse lines into sections based on headers
                for line in lines:
                    line_strip = line.strip()
                    if line_strip in headers:
                        current_section = line_strip
                        sections[current_section] = []
                    elif current_section and line_strip.startswith('* '):
                         sections[current_section].append(line_strip[2:].strip()) # Store content after '* '
                    elif current_section and line_strip and not line_strip.startswith("---"): # Capture potential continuation lines or 'None found.'
                         # Check if it's the "None found" message
                         if "none found" in line_strip.lower():
                             sections[current_section].append(line_strip)
                         # If the list is not empty and last item wasn't "None found", append as continuation
                         elif sections[current_section] and "none found" not in sections[current_section][-1].lower():
                             sections[current_section][-1] += " " + line_strip
                         else: # Otherwise, treat as a new item (or handle as unexpected format)
                              sections[current_section].append(line_strip)


                # Display parsed sections
                for header in headers:
                    st.markdown(f'<p class="summary-header">{header.replace("#### ", "")}</p>', unsafe_allow_html=True)
                    if header in sections and sections[header]:
                        for item in sections[header]:
                            if "none found" in item.lower():
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*{item}*") # Indent and italicize 'None found'
                            elif item.startswith('+'):
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span class='summary-add'>+ {item[1:].strip()}</span>", unsafe_allow_html=True)
                            elif item.startswith('-'):
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span class='summary-del'>- {item[1:].strip()}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{item}") # Display unexpected format as is
                    else:
                         st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;*None found.*") # Default if section missing


            except Exception as e:
                st.error(f"Failed to parse AI summary format. Displaying raw output:\nError: {e}")
                st.markdown(f"```markdown\n{summary_text}\n```") # Fallback to code block


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

