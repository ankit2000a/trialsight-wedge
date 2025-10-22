import streamlit as st
import fitz  # PyMuPDF
import difflib
import google.generativeai as genai
import os
import time
import re
import unicodedata # For robust normalization
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
    # Using only one normalized text version now
    keys = ['file1_data', 'file2_data', 'diff_html', 'summary',
            'original_text_normalized', 'revised_text_normalized',
            'processing_comparison']
    for key in keys:
        if key not in st.session_state: st.session_state[key] = None
    if st.session_state.processing_comparison is None: st.session_state.processing_comparison = False
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
             try: st.session_state.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
             except Exception: st.session_state.gemini_model = genai.GenerativeModel('gemini-pro') # Fallback
        model = st.session_state.gemini_model
        ai_enabled = True
    else: st.warning("Google API Key not found. AI Summary disabled."); ai_enabled = False
except Exception as e:
    ai_enabled = False; st.warning(f"Could not initialize Google AI: {e}")
    if 'gemini_model' in st.session_state: st.session_state.gemini_model = None

# --- Helper Functions ---

# --- CHATGPT'S NORMALIZATION FUNCTION ---
@st.cache_data
def normalize_pdf_text(file_bytes, filename="file"):
    """
    Extracts and normalizes text preserving line structure, joining broken words.
    Based on user-provided ChatGPT suggestion.
    """
    try:
        # Extract raw text first
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text", sort=True) for page in doc) # Join pages with newline

        if not text:
             print(f"Warning: No raw text extracted from {filename}")
             return f"ERROR: No text found in PDF {filename}"

        # 1. Normalize weird Unicode (NFKC handles ligatures, etc.)
        text = unicodedata.normalize("NFKC", text)

        # 2. Fix words broken by a newline (Pu\nblishers → Publishers)
        #    Only join if both sides seem like word characters
        text = re.sub(r'([a-zA-Z0-9])\n([a-zA-Z0-9])', r'\1\2', text)

        # 3. Fix hyphenated words broken by newline (state-\n R\N? -> \N )
        text = re.sub(r'-\n', '', text) # Remove newline immediately after hyphen

        # 4. Replace multiple spaces/tabs with a single space (preserves newlines for now)
        text = re.sub(r'[ \t]+', ' ', text)

        # 5. Normalize line endings (essential before splitting)
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 6. Remove leading/trailing spaces on each line
        lines = text.split('\n')
        stripped_lines = [line.strip() for line in lines]

        # 7. Remove blank lines (lines that become empty after stripping)
        non_blank_lines = [line for line in stripped_lines if line]
        text = '\n'.join(non_blank_lines)

        # 8. Final strip for the whole text block
        text = text.strip()

        if not text:
             print(f"Warning: Text became empty after normalization for {filename}")
             # Return error if normalization made it empty, might indicate issue
             return f"ERROR: Text became empty after normalization for {filename}"
        return text

    except Exception as e:
        print(f"Error normalizing {filename}: {e}")
        return f"ERROR: Could not read/normalize {filename}. Details: {e}"


def generate_diff_html(text1_norm, text2_norm, filename1="Original", filename2="Revised"):
    """Creates side-by-side HTML diff using the NORMALIZED text (preserving lines)."""
    if text1_norm is None or text2_norm is None or text1_norm.startswith("ERROR:") or text2_norm.startswith("ERROR:"):
         return "Error: Cannot generate visual diff (text normalization failed)."
    try:
        # Use the normalized text directly, split by its preserved newlines
        lines1 = text1_norm.splitlines()
        lines2 = text2_norm.splitlines()

        html = difflib.HtmlDiff(wrapcolumn=100, tabsize=4).make_table(
            lines1,
            lines2,
            fromdesc=filename1 + " (Normalized)",
            todesc=filename2 + " (Normalized)"
        )
        style = f"<style>{difflib.HtmlDiff._styles}</style>"
        custom_style = """
        <style>
        table.diff { font-family: Consolas, 'Courier New', monospace; border-collapse: collapse; width: 100%; font-size: 0.875em; table-layout: fixed; }
        .diff_header { background-color: #374151; color: #E5E7EB; padding: 0.2em 0.5em; font-weight: bold; position: sticky; top: 0; z-index: 10;}
        td { padding: 0.1em 0.4em; vertical-align: top; white-space: pre-wrap; word-wrap: break-word; }
        .diff_next { background-color: #4b5563; }
        .diff_add { background-color: rgba(16, 185, 129, 0.1); }
        .diff_chg { background-color: rgba(209, 163, 23, 0.1); }
        .diff_sub { background-color: rgba(239, 68, 68, 0.1); text-decoration: none; border-bottom: 1px dotted rgba(239, 68, 68, 0.5); }
        </style>
        """
        return f"<div style='overflow-x: auto;'>{style + custom_style + html}</div>"
    except Exception as e:
        print(f"Error generating HTML diff: {e}")
        if isinstance(e, RecursionError):
            return "Error: Failed to generate visual comparison (text complexity)."
        return f"Error: Failed to generate visual comparison. {e}"


def get_ai_summary(text1_norm, text2_norm):
    """Generates a categorized summary using AI based on normalized diff_match_patch comparison."""
    # --- Input Validation ---
    if not ai_enabled: return "AI Summary feature is not available (API key issue)."
    if text1_norm is None or text2_norm is None or \
       (isinstance(text1_norm, str) and text1_norm.startswith("ERROR:")) or \
       (isinstance(text2_norm, str) and text2_norm.startswith("ERROR:")):
        return "AI Summary cannot be generated: text normalization failed."

    # --- Use diff_match_patch on the CORRECTLY NORMALIZED text ---
    dmp = dmp_module.diff_match_patch()
    dmp.Diff_Timeout = 1.0 # Keep timeout
    try:
        # Pass the normalized text (which still has structure) to DMP
        diffs = dmp.diff_main(text1_norm, text2_norm)
        dmp.diff_cleanupSemantic(diffs) # Improves readability
    except Exception as dmp_err:
         print(f"Error during diff_match_patch: {dmp_err}")
         if text1_norm == text2_norm: return "INFO: No textual differences found after robust normalization."
         else: return f"ERROR: Failed to compute differences using diff_match_patch: {dmp_err}."

    # --- Extract ONLY true additions and deletions for the AI ---
    meaningful_diff_fragments_for_ai = []
    for op, data in diffs:
        # Normalize whitespace within the fragment for cleaner AI input, then strip
        data_clean = re.sub(r'\s+', ' ', data).strip()
        if not data_clean: continue # Skip empty/whitespace fragments

        prefix = ""
        if op == dmp.DIFF_INSERT: prefix = "+"
        elif op == dmp.DIFF_DELETE: prefix = "-"
        else: continue # Skip equal parts

        meaningful_diff_fragments_for_ai.append(f"{prefix}{data_clean}\n") # Add newline separator


    # --- Handle No Differences Case ---
    if not meaningful_diff_fragments_for_ai:
         if text1_norm == text2_norm:
            return "INFO: No textual differences were found between the documents after robust normalization."
         else:
             # If texts weren't identical but DMP found no diffs (e.g., timeout, edge case)
             return "INFO: No significant content changes detected by diff_match_patch after robust normalization. Minor variations might exist."


    diff_text_for_prompt = "".join(meaningful_diff_fragments_for_ai)

    # --- AI Prompt (v17 - Using correctly normalized DMP diff fragments) ---
    # Keep the same prompt asking for 3 categories
    prompt = f"""
    Analyze the provided ADDED (+) and DELETED (-) content fragments. These fragments represent meaningful content changes between two clinical trial protocol versions after robust normalization (lowercase, rejoined words, normalized whitespace). Categorize these fragments.

    **Instructions:**
    1.  **Clinically Significant Changes:** Identify ADDED (+) or DELETED (-) fragments clearly related to:
        * Inclusion/Exclusion criteria
        * Dosage / Treatment schedules
        * Procedures / Assessments
        * Safety reporting
        * Objectives / Endpoints
        List these first. Indicate the nature of the change (added/deleted content).
    2.  **Other Added Content:** List ALL OTHER provided ADDED (+) fragments.
    3.  **Other Deleted Content:** List ALL OTHER provided DELETED (-) fragments.
    4.  **Output Format:** Structure your response EXACTLY like this, using markdown H4 (####) for headers:

        #### Clinically Significant Changes (Added/Deleted Content)
        * [List ONLY significant ADDED (+) or DELETED (-) content here. If none found, state "None found."]

        #### Other Added Content
        * [List ALL OTHER non-blank ADDED (+) content from the input here. If none found, state "None found."]

        #### Other Deleted Content
        * [List ALL OTHER non-blank DELETED (-) content from the input here. If none found, state "None found."]

    **Meaningful Added (+) and Deleted (-) Content Fragments (Robustly Normalized):**
    ---
    {diff_text_for_prompt[:8000]}
    ---

    Begin Summary:
    """


    try:
        # Safety settings and generation config remain the same
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
             safety_ratings_str = "N/A"
             if finish_reason == 'SAFETY' and hasattr(candidate, 'safety_ratings'): safety_ratings_str = ", ".join([f"{r.category.name}: {r.probability.name}" for r in candidate.safety_ratings])
             return f"ERROR: AI model returned an empty response. Finish Reason: {finish_reason}. Safety Ratings: [{safety_ratings_str}]"
    except Exception as e:
        error_message = f"ERROR: Failed to get AI summary: {e}"
        if "quota" in str(e).lower() or "429" in str(e): error_message += "\n(Quota exceeded?)"
        # Return error string for display
        return error_message

# --- Main App UI ---
st.title("📄 TrialSight: Document Comparator")
st.markdown("Compare two versions of a document and get an AI-powered summary of content changes.")
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
            keys_to_clear = ['diff_html', 'summary', 'original_text_normalized', 'revised_text_normalized', 'processing_comparison']
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
        keys_to_clear = ['file1_data', 'file2_data', 'diff_html', 'summary', 'original_text_normalized', 'revised_text_normalized', 'processing_comparison']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

# --- Comparison Logic ---
if st.session_state.get('file1_data') and st.session_state.get('file2_data'):
    # Show Compare button only if needed
    if not st.session_state.get('diff_html') and not st.session_state.get('processing_comparison'):
        if st.button("Compare Documents", type="primary", use_container_width=True):
            st.session_state.processing_comparison = True
            # Clear previous results
            keys_to_clear = ['diff_html', 'summary', 'original_text_normalized', 'revised_text_normalized']
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

                # Generate HTML diff using normalized text (handles internal errors)
                st.session_state['diff_html'] = generate_diff_html(text1_norm, text2_norm, file1.name, file2.name)
            else:
                st.session_state['diff_html'] = "Error: File data missing."
                st.session_state['summary'] = None
            st.session_state.processing_comparison = False # Reset flag
            # Rerun only if diff generation didn't return an error string
            if isinstance(st.session_state.get('diff_html'), str) and not st.session_state.get('diff_html', "").startswith("Error:"):
                 st.rerun()


# --- Display Results Section ---
if not st.session_state.get('processing_comparison') and st.session_state.get('diff_html') and isinstance(st.session_state.get('diff_html'), str) and not st.session_state.get('diff_html', "").startswith("Error:"):

    # --- DEBUGGER (Shows the robustly normalized text) ---
    with st.expander("Show/Hide Normalized Text (Used for Comparison)"):
        col1, col2 = st.columns(2)
        original_display = st.session_state.get('original_text_normalized') if isinstance(st.session_state.get('original_text_normalized'), str) else "Normalization Error or Not Run"
        revised_display = st.session_state.get('revised_text_normalized') if isinstance(st.session_state.get('revised_text_normalized'), str) else "Normalization Error or Not Run"
        with col1: st.subheader("Original (Normalized)"); st.code(original_display, language=None)
        with col2: st.subheader("Revised (Normalized)"); st.code(revised_display, language=None)

    # --- Side-by-Side Diff (using normalized text) ---
    st.subheader("Visual Comparison (Based on Normalized Text)")
    st.markdown("*(Note: Original formatting may differ slightly due to normalization)*") # Updated note
    st.components.v1.html(st.session_state.diff_html, height=400, scrolling=True)


    st.markdown("---")

    # --- AI Summary ---
    st.subheader("🤖 AI-Powered Summary of Content Changes")
    st.markdown("Click for summary based on normalized text comparison.")
    # Disable button checks based on normalized text versions
    button_disabled = not ai_enabled or (isinstance(st.session_state.get('original_text_normalized'), str) and st.session_state.get('original_text_normalized', "").startswith("ERROR:")) \
                      or (isinstance(st.session_state.get('revised_text_normalized'), str) and st.session_state.get('revised_text_normalized', "").startswith("ERROR:")) \
                      or st.session_state.get('original_text_normalized') is None or st.session_state.get('revised_text_normalized') is None

    summary_button_label = "✨ Get Content Changes Summary"
    if st.session_state.get('summary') is not None: summary_button_label = "🔄 Regenerate Summary"

    if st.button(summary_button_label, use_container_width=True, disabled=button_disabled, key="gen_summary_btn"):
        # Use the ROBUSTLY NORMALIZED texts stored in state
        if st.session_state.get('original_text_normalized') is not None and st.session_state.get('revised_text_normalized') is not None:
            with st.spinner("Analyzing content changes..."):
                summary_result = get_ai_summary(st.session_state.original_text_normalized, st.session_state.revised_text_normalized)
                st.session_state['summary'] = summary_result
                st.rerun()
        else: st.error("Cannot generate summary: Normalized text missing.")

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
                # Parsing logic remains the same
                sections = {}
                current_section_key = None
                headers_map = { # Ensure these match the AI prompt's H4 headers
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
                    if current_section_key and line_strip.startswith('* '):
                         item_text = line_strip[2:].strip()
                         sections[current_section_key].append(item_text)
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
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span class='summary-add'>{item}</span>", unsafe_allow_html=True)
                                elif item.startswith('-'):
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span class='summary-del'>{item}</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{item}")
                    else:
                         st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;*None found.*") # Display None found if section is missing or truly empty
            except Exception as e:
                st.error(f"Failed to parse AI summary format. Raw output:\nError: {e}")
                st.markdown(f"```markdown\n{summary_text}\n```") # Fallback to raw

    # Explain disabled button state
    elif button_disabled and not st.session_state.get('processing_comparison'):
         if not ai_enabled: st.warning("AI Summary disabled: API Key missing/invalid.")
         # Check normalized text versions for failure
         elif (isinstance(st.session_state.get('original_text_normalized'), str) and st.session_state.get('original_text_normalized', "").startswith("ERROR:")) or \
              (isinstance(st.session_state.get('revised_text_normalized'), str) and st.session_state.get('revised_text_normalized', "").startswith("ERROR:")):
             st.warning("AI Summary disabled: Text normalization failed.")
         elif st.session_state.get('original_text_normalized') is None or st.session_state.get('revised_text_normalized') is None:
             st.warning("AI Summary disabled: Click 'Compare Documents' first.")

# Handle Errors / Loading State
elif st.session_state.get('diff_html') and isinstance(st.session_state.get('diff_html'), str) and st.session_state.get('diff_html', "").startswith("Error:"):
    st.error(st.session_state.diff_html)
elif st.session_state.get('processing_comparison'):
     st.markdown('<div class="loader-container"><div class="loader"></div><p>Processing...</p></div>', unsafe_allow_html=True)

