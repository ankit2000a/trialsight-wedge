import streamlit as st
import fitz  # PyMuPDF
# import difflib # Using DMP now
import os
import time
import re
import unicodedata # For robust normalization
# --- ENSURE THESE ARE IMPORTED ---
try:
    import google.generativeai as genai
except ImportError:
    st.error("ERROR: google.generativeai library not found. Please install it: pip install google-generativeai")
    # Set dummy genai object to avoid further NameErrors downstream if import fails
    class DummyGenAI: pass
    genai = DummyGenAI()
    genai.GenerativeModel = lambda x: None # Mock the model function
    api_key = None # Ensure api_key is None if import fails
    ai_enabled = False

try:
    import diff_match_patch as dmp_module
except ImportError:
    st.error("ERROR: diff-match-patch library not found. Please install it: pip install diff-match-patch")
    # Set dummy dmp object
    class DummyDMP:
        def diff_main(self, t1, t2): return []
        def diff_cleanupSemantic(self, d): pass
        def diff_prettyHtml(self, d): return "Error: diff-match-patch not installed."
        DIFF_INSERT = 1
        DIFF_DELETE = -1
        DIFF_EQUAL = 0
    dmp_module = DummyDMP()


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
    /* Make uploader less intrusive when files are loaded */
    .stFileUploader > div:has(button[kind="secondary"]) { border: none; padding: 0; }
    .stFileUploader > div:has(button[kind="secondary"]) > div { padding-top: 0; } /* Reduce top padding */
    /* Hide default upload text/button when files ARE loaded */
    .stFileUploader [data-testid="stFileUploadDropzone"]:has(+ div [data-testid="stFileUploaderFile"]) button { display: none; }
    .stFileUploader [data-testid="stFileUploadDropzone"]:has(+ div [data-testid="stFileUploaderFile"]) p { display: none; }


    .file-card { background-color: #2D3748; border-radius: 0.5rem; padding: 0.8rem 1rem; border: 1px solid #4A5568; display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;}
    /* Diff (using DMP prettyHTML styles) */
    .diff-container { font-family: Consolas, 'Courier New', monospace; font-size: 0.9em; line-height: 1.4; border: 1px solid #4A5568; border-radius: 0.5rem; padding: 1rem; background-color: #1A202C; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }
    ins { background-color: rgba(16, 185, 129, 0.2); color: #A7F3D0; text-decoration: none; }
    del { background-color: rgba(239, 68, 68, 0.2); color: #FECACA; text-decoration: none; }
    /* Summary Styles */
    .summary-add { color: #6EE7B7; }
    .summary-del { color: #FCA5A5; text-decoration: line-through; }
    .summary-header { font-weight: bold; margin-top: 0.8em; margin-bottom: 0.3em; color: #9CA3AF; } /* Header color */
    .summary-item { margin-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.3em;} /* Hanging indent */
    .summary-context { font-style: italic; color: #6B7280; font-size: 0.9em; } /* Context style */

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
    keys = ['file1_data', 'file2_data', 'diff_html_output', 'summary',
            'original_text_normalized', 'revised_text_normalized',
            'processing_comparison']
    for key in keys:
        if key not in st.session_state: st.session_state[key] = None
    if st.session_state.processing_comparison is None: st.session_state.processing_comparison = False
init_session_state()

# --- Gemini API Configuration ---
# Moved after imports, includes fallback logic
ai_enabled = False
api_key = None
# Check if genai was imported successfully before trying to use it
if 'google.generativeai' in sys.modules:
    try:
        api_key_secret = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets else None
        api_key_env = os.environ.get("GOOGLE_API_KEY")
        api_key = api_key_secret or api_key_env
        if api_key:
            genai.configure(api_key=api_key)
            if 'gemini_model' not in st.session_state or st.session_state.gemini_model is None:
                 try: st.session_state.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
                 except Exception:
                      st.warning("Failed to initialize gemini-2.5-pro, falling back to gemini-pro.")
                      st.session_state.gemini_model = genai.GenerativeModel('gemini-pro') # Fallback
            model = st.session_state.gemini_model # Assign model from state
            ai_enabled = True # Enable only if key and model init succeed
        else:
            st.warning("Google API Key not found. AI Summary disabled.")
            ai_enabled = False
    except Exception as e:
        ai_enabled = False; st.warning(f"Could not initialize Google AI: {e}")
        if 'gemini_model' in st.session_state: st.session_state.gemini_model = None # Reset model state on error
# else: ai_enabled is already False from import failure


# --- Helper Functions ---

# --- USING PRECISE CHATGPT NORMALIZATION FUNCTION ---
@st.cache_data
def normalize_pdf_text(file_bytes, filename="file"):
    """
    Extracts and normalizes text preserving line structure, joining broken words.
    Based on user-provided ChatGPT suggestion. Handles PDF or TXT input.
    """
    text = ""
    is_pdf = filename.lower().endswith('.pdf')
    try:
        if is_pdf:
            # Added check for empty file bytes
            if not file_bytes: return f"ERROR: Uploaded PDF file '{filename}' is empty."
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page_texts = [page.get_text("text", sort=True) or "" for page in doc]
            text = "\n".join(page_texts)
            doc.close()
        else: # Handle plain text file
             # Added check for empty file bytes
             if not file_bytes: return f"ERROR: Uploaded TXT file '{filename}' is empty."
             text = file_bytes.decode('utf-8', errors='ignore') # Ignore decoding errors

        if not text:
             # This check might be redundant if the above catches empty files, but keep for safety
             print(f"Warning: No raw text extracted from {filename}")
             return f"ERROR: No text found in file {filename}"

        # 1. Normalize weird Unicode (NFKC handles ligatures, etc.)
        text = unicodedata.normalize("NFKC", text)

        # 2. Fix words broken by a newline (Pu\nblishers → Publishers)
        #    Only join if both sides seem like word characters, handle optional space
        text = re.sub(r'([a-zA-Z0-9])\s*\n\s*([a-zA-Z0-9])', r'\1\2', text)

        # 3. Fix hyphenated words broken by newline (state-\n -> state-)
        text = re.sub(r'-\s*\n\s*', '', text) # Remove newline and surrounding space after hyphen

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

        # 9. DO NOT CONVERT TO LOWERCASE HERE - keep case for visual diff
        # text_lower = text.lower() # Remove this


        if not text:
             print(f"Warning: Text became empty after normalization for {filename}")
             return f"ERROR: Text became empty after normalization for {filename}"
        # Return ONLY the normalized text with original case
        return text

    except fitz.fitz.FileDataError as fe:
         print(f"Fitz FileDataError for {filename}: {fe}")
         return f"ERROR: Could not read PDF '{filename}'. File might be corrupted or password-protected."
    except Exception as e:
        print(f"Error normalizing {filename}: {e}")
        return f"ERROR: Could not read/normalize {filename}. Details: {e}"


# --- Use diff-match-patch for HTML ---
def generate_dmp_diff_html(text1_norm, text2_norm):
    """Creates highlighted HTML diff using diff-match-patch on normalized text."""
    if text1_norm is None or text2_norm is None or \
       text1_norm.startswith("ERROR:") or text2_norm.startswith("ERROR:"):
         return "<p style='color:red;'>Error: Cannot generate visual diff (text normalization failed).</p>"
    try:
        # Check if dmp_module was imported correctly
        if not hasattr(dmp_module, 'diff_match_patch'):
             return "<p style='color:red;'>Error: diff-match-patch library not loaded correctly.</p>"

        dmp = dmp_module.diff_match_patch()
        dmp.Diff_Timeout = 2.0
        # Use normalized text (original case, preserved lines) for visual diff
        diffs = dmp.diff_main(text1_norm, text2_norm)
        dmp.diff_cleanupSemantic(diffs) # Makes diff more human-readable

        html = dmp.diff_prettyHtml(diffs)
        # Wrap in styled container
        return f"<div class='diff-container'>{html}</div>"
    except Exception as e:
        print(f"Error generating DMP HTML diff: {e}")
        return f"<p style='color:red;'>Error: Failed to generate visual comparison using diff-match-patch. {e}</p>"


def get_ai_summary(text1_norm, text2_norm):
    """Generates a categorized summary using AI based on normalized diff_match_patch comparison."""
    # --- Input Validation ---
    if not ai_enabled: return "AI Summary feature is not available (API key issue)."
    # Check the normalized text versions
    if text1_norm is None or text2_norm is None or \
       (isinstance(text1_norm, str) and text1_norm.startswith("ERROR:")) or \
       (isinstance(text2_norm, str) and text2_norm.startswith("ERROR:")):
        return "AI Summary cannot be generated: text normalization failed."

    # --- Use diff_match_patch on the CORRECTLY NORMALIZED text ---
    # Convert to lowercase SPECIFICALLY for the AI comparison step to ignore case
    text1_lower = text1_norm.lower()
    text2_lower = text2_norm.lower()

    # Check if dmp_module was imported correctly
    if not hasattr(dmp_module, 'diff_match_patch'):
         return "ERROR: diff-match-patch library not loaded correctly."

    dmp = dmp_module.diff_match_patch()
    dmp.Diff_Timeout = 1.0
    try:
        diffs = dmp.diff_main(text1_lower, text2_lower) # Use lowercase versions for AI input diff
        dmp.diff_cleanupSemantic(diffs)
    except Exception as dmp_err:
         print(f"Error during diff_match_patch: {dmp_err}")
         if text1_lower == text2_lower: return "INFO: No textual differences found after robust normalization."
         else: return f"ERROR: Failed to compute differences using diff_match_patch: {dmp_err}."

    # --- Extract ONLY true additions and deletions for the AI ---
    meaningful_diff_fragments_for_ai = []
    MIN_FRAGMENT_LEN = 3 # Filter very short fragments (often noise)
    for op, data in diffs:
        # Normalize whitespace WITHIN the fragment for cleaner AI input, then strip
        data_clean = re.sub(r'\s+', ' ', data).strip()
        if not data_clean or len(data_clean) < MIN_FRAGMENT_LEN: continue

        prefix = ""
        # Check op against dmp constants
        if op == dmp_module.DIFF_INSERT: prefix = "+"
        elif op == dmp_module.DIFF_DELETE: prefix = "-"
        else: continue # Skip equal parts

        meaningful_diff_fragments_for_ai.append(f"{prefix}{data_clean}\n") # Add newline separator


    # --- Handle No Differences Case ---
    if not meaningful_diff_fragments_for_ai:
         if text1_lower == text2_lower:
            return "INFO: No textual differences were found between the documents after robust normalization."
         else:
             # If texts weren't identical but no fragments met criteria
             return "INFO: No significant content changes detected by diff_match_patch after robust normalization. Only minor variations might exist."


    diff_text_for_prompt = "".join(meaningful_diff_fragments_for_ai)

    # --- AI Prompt (v23 - Using correctly normalized DMP diff fragments, focus on content) ---
    prompt = f"""
    Analyze the provided ADDED (+) and DELETED (-) content fragments from a comparison of two normalized clinical trial protocols (lowercase, rejoined words, normalized whitespace). Categorize these fragments based on potential clinical significance, ignoring minor variations.

    **Instructions:**
    1.  **Focus on Meaning:** Identify ADDED (+) or DELETED (-) fragments that represent a *substantive change* in meaning or requirements.
    2.  **Clinically Significant:** Prioritize changes clearly related to: Inclusion/Exclusion criteria, Dosage/Treatment, Procedures/Assessments, Safety reporting, or Objectives/Endpoints. List these first with brief context.
    3.  **Other Substantive Changes:** List other ADDED/DELETED fragments that change the actual content (not just trivial wording differences).
    4.  **IGNORE Noise:** Explicitly IGNORE fragments that are likely just minor rephrasing, single common word substitutions (a/the), isolated numbers/symbols, or artifacts of normalization if they don't change the core meaning.
    5.  **Output Format:** Structure your response EXACTLY like this:

        #### Clinically Significant Changes (Added/Deleted Content)
        * [List ONLY significant ADDED (+) or DELETED (-) content here, with brief justification. If none found, state "None found."]

        #### Other Substantive Added Content
        * [List ALL OTHER clearly ADDED (+) content fragments here. If none found, state "None found."]

        #### Other Substantive Deleted Content
        * [List ALL OTHER clearly DELETED (-) content fragments here. If none found, state "None found."]

    **Meaningful Added (+) and Deleted (-) Content Fragments (Robustly Normalized):**
    ---
    {diff_text_for_prompt[:8000]}
    ---

    Begin Summary:
    """


    try:
        # Safety settings and generation config remain the same
        safety_settings = [ {"category": c.name, "threshold": "BLOCK_LOW_AND_ABOVE"} for c in genai.types.HarmCategory] # Use enum names
        generation_config = genai.types.GenerationConfig(temperature=0.1)
        if 'gemini_model' not in st.session_state or st.session_state.gemini_model is None:
             return "ERROR: Gemini model not initialized."
        model = st.session_state.gemini_model
        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)

        # Robust response handling (unchanged)
        if not response.candidates:
            block_reason = "Unknown";
            if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'): block_reason = response.prompt_feedback.block_reason.name # Use enum name
            return f"ERROR: AI response blocked. Reason: {block_reason}."
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason.name if hasattr(candidate, 'finish_reason') else "Unknown" # Use enum name
        if candidate.content and candidate.content.parts and candidate.content.parts[0].text:
             response_text = candidate.content.parts[0].text.strip()
             return response_text
        else:
             safety_ratings_str = "N/A"
             # Updated safety rating access
             if finish_reason == 'SAFETY' and hasattr(candidate, 'safety_ratings'):
                  safety_ratings_str = ", ".join([f"{r.category.name}: {r.probability.name}" for r in candidate.safety_ratings])
             return f"ERROR: AI model returned an empty response. Finish Reason: {finish_reason}. Safety Ratings: [{safety_ratings_str}]"
    except Exception as e:
        error_message = f"ERROR: Failed to get AI summary: {e}"
        if "quota" in str(e).lower() or "429" in str(e): error_message += "\n(Quota exceeded?)"
        # Also check for API key validation errors
        elif "api key not valid" in str(e).lower(): error_message += "\n(Invalid API Key? Check secrets.toml)"
        return error_message

# --- Main App UI ---
st.title("📄 TrialSight: Document Content Comparator")
st.markdown("Highlights content changes between two documents using robust text normalization.")
st.markdown("---")

# File Uploader & Display Logic
# Use columns for better layout after files uploaded
uploader_col, clear_col = st.columns([0.85, 0.15])

with uploader_col:
    if not st.session_state.get('file1_data') or not st.session_state.get('file2_data'):
        # Only show uploader if files are not loaded
        uploaded_files = st.file_uploader(
            "Upload Original & Revised Files (PDF or TXT)",
            type=["pdf", "txt"], # Allow TXT
            accept_multiple_files=True,
            key="file_uploader"
            )
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
                st.rerun() # Rerun immediately after upload
    else:
        # Display file cards if files are loaded
        col1, col2 = st.columns(2)
        file1_name = st.session_state.file1_data.name if st.session_state.get('file1_data') else "N/A"
        file2_name = st.session_state.file2_data.name if st.session_state.get('file2_data') else "N/A"
        with col1:
             st.markdown(f"""
             <div class="file-card">
                 <span>Original: <strong>{file1_name}</strong></span>
             </div>
             """, unsafe_allow_html=True)
        with col2:
             st.markdown(f"""
             <div class="file-card">
                 <span>Revised: <strong>{file2_name}</strong></span>
             </div>
             """, unsafe_allow_html=True)

# Place clear button separately, ensure it's visible when files are loaded
with clear_col:
     if st.session_state.get('file1_data') and st.session_state.get('file2_data'):
        # Add some top margin to align better with file cards
        st.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True) # Adjusted margin
        if st.button("Clear", key="clear_btn", use_container_width=True):
            keys_to_clear = ['file1_data', 'file2_data', 'diff_html_output', 'summary', 'original_text_normalized', 'revised_text_normalized', 'processing_comparison']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            st.rerun()


# --- Comparison Logic ---
if st.session_state.get('file1_data') and st.session_state.get('file2_data'):
    # Show Compare button only if needed (results not present AND not processing)
    if not st.session_state.get('diff_html_output') and not st.session_state.get('processing_comparison'):
        if st.button("Compare Documents", type="primary", use_container_width=True):
            st.session_state.processing_comparison = True
            # Clear previous results explicitly on button press
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
                # Determine file types for normalization function
                file1_type = file1.name.split('.')[-1].lower()
                file2_type = file2.name.split('.')[-1].lower()

                # --- Extract NORMALIZED text using the CORRECT function ---
                # Changed function name here to use the new one
                text1_norm = normalize_pdf_text(file1_bytes, file1.name) # Use the correct function
                text2_norm = normalize_pdf_text(file2_bytes, file2.name)

                # Store normalized text (original case version for display)
                st.session_state['original_text_normalized'] = text1_norm
                st.session_state['revised_text_normalized'] = text2_norm

                # Generate HTML diff using normalized text (handles internal errors)
                st.session_state['diff_html_output'] = generate_dmp_diff_html(text1_norm, text2_norm)
            else:
                st.session_state['diff_html_output'] = "<p style='color:red;'>Error: File data missing.</p>"
                st.session_state['summary'] = None
            st.session_state.processing_comparison = False # Reset flag
            # Rerun regardless of diff success to show either diff or error
            st.rerun()


# --- Display Results Section ---
if not st.session_state.get('processing_comparison') and st.session_state.get('diff_html_output'):

     # Check if diff generation itself resulted in an error message
    # Handle both None and error strings robustly
    diff_output = st.session_state.diff_html_output
    is_diff_error = diff_output is None or (isinstance(diff_output, str) and diff_output.strip().lower().startswith("<p style='color:red;'>error:"))

    if is_diff_error:
        st.error("Failed to generate visual comparison:")
        # Display the HTML error or a default message if None
        st.markdown(diff_output if diff_output else "<p style='color:red;'>Unknown error generating comparison.</p>", unsafe_allow_html=True)
    else:
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
        st.markdown("*(Green = Added, Red = Deleted. Formatting differences ignored due to normalization)*")
        st.markdown(st.session_state.diff_html_output, unsafe_allow_html=True) # Display DMP HTML

        st.markdown("---")

        # --- AI Summary ---
        st.subheader("🤖 AI-Powered Summary of Content Changes")
        st.markdown("Click for summary based on normalized/filtered comparison.")
        # Disable button checks based on normalized text versions
        # Check for None AND error strings now
        norm_text_ok = True
        if st.session_state.get('original_text_normalized') is None or (isinstance(st.session_state.get('original_text_normalized'), str) and st.session_state.get('original_text_normalized', "").startswith("ERROR:")): norm_text_ok = False
        if st.session_state.get('revised_text_normalized') is None or (isinstance(st.session_state.get('revised_text_normalized'), str) and st.session_state.get('revised_text_normalized', "").startswith("ERROR:")): norm_text_ok = False

        button_disabled = not ai_enabled or not norm_text_ok


        summary_button_label = "✨ Get Content Changes Summary"
        if st.session_state.get('summary') is not None: summary_button_label = "🔄 Regenerate Summary"

        if st.button(summary_button_label, use_container_width=True, disabled=button_disabled, key="gen_summary_btn"):
            # Use the ROBUSTLY NORMALIZED texts stored in state
            if norm_text_ok:
                with st.spinner("Analyzing content changes..."):
                    summary_result = get_ai_summary(st.session_state.original_text_normalized, st.session_state.revised_text_normalized)
                    st.session_state['summary'] = summary_result
                    st.rerun()
            else: st.error("Cannot generate summary: Normalized text missing or invalid.")

        # --- Display Summary (Using Formatted Display Logic - unchanged) ---
        if st.session_state.get('summary') is not None:
             summary_text = st.session_state.summary
             st.markdown("---") # Add separator
             # Check for specific error/info prefixes
             if summary_text.startswith("ERROR:") or "cannot be generated" in summary_text or "not available" in summary_text:
                 st.error(summary_text) # Display as error
             elif summary_text.startswith("INFO:") or "No textual differences" in summary_text or "No substantive" in summary_text:
                  st.info(summary_text.replace("INFO: ", "")) # Display as info
             else:
                # Attempt to parse and display the formatted summary
                try:
                    sections = {}
                    current_section_key = None
                    headers_map = { # Ensure these match the AI prompt's H4 headers
                        "#### Clinically Significant Changes (Added/Deleted Content)": "significant",
                        "#### Other Substantive Added Content": "added", # Match prompt
                        "#### Other Substantive Deleted Content": "deleted" # Match prompt
                    }
                    header_display = {
                         "significant": "Clinically Significant Changes (Added/Deleted Content)",
                         "added": "Other Substantive Added Content",
                         "deleted": "Other Substantive Deleted Content"
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
                        # Add item to current section if format matches
                        if current_section_key and line_strip.startswith('* '):
                             item_text = line_strip[2:].strip()
                             sections[current_section_key].append(item_text)
                        elif current_section_key and line_strip and "none found" in line_strip.lower(): # Capture 'None found' case
                             sections[current_section_key].append(line_strip)

                    st.markdown("### Categorized Summary of Content Changes:") # Display title
                    # Display each section
                    for key, display_name in header_display.items():
                        st.markdown(f'<p class="summary-header">{display_name}</p>', unsafe_allow_html=True)
                        if key in sections and sections[key]:
                            items = sections[key]
                            # Handle 'None found' display
                            if len(items) == 1 and "none found" in items[0].lower():
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*{items[0]}*")
                            else:
                                # Display actual list items with styling
                                for item in items:
                                    item_html = item # Start with raw item text
                                    # Basic check for +/- prefix for styling
                                    if item.startswith('+'):
                                        item_html = f"<span class='summary-add'>{item}</span>"
                                    elif item.startswith('-'):
                                        item_html = f"<span class='summary-del'>{item}</span>"
                                    # Wrap in div for consistent indentation/layout
                                    st.markdown(f"<div class='summary-item'>{item_html}</div>", unsafe_allow_html=True)
                        else:
                             st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;*None found.*") # If section key wasn't found or list was empty

                except Exception as e:
                    # Fallback to raw display if parsing fails
                    st.error(f"Failed to parse AI summary format. Raw output:\nError: {e}")
                    st.markdown(f"```markdown\n{summary_text}\n```")

        # Explain disabled button state if summary hasn't been generated
        elif button_disabled and not st.session_state.get('processing_comparison'):
             if not ai_enabled: st.warning("AI Summary disabled: API Key missing/invalid.")
             # Check normalized text versions for failure
             elif (isinstance(st.session_state.get('original_text_normalized'), str) and st.session_state.get('original_text_normalized', "").startswith("ERROR:")) or \
                  (isinstance(st.session_state.get('revised_text_normalized'), str) and st.session_state.get('revised_text_normalized', "").startswith("ERROR:")):
                 st.warning("AI Summary disabled: Text normalization failed.")
             elif st.session_state.get('original_text_normalized') is None or st.session_state.get('revised_text_normalized') is None:
                 st.warning("AI Summary disabled: Click 'Compare Documents' first.")

# Handle Errors / Loading State
# Display error from generate_dmp_diff_html if it occurred
elif not st.session_state.get('processing_comparison') and st.session_state.get('diff_html_output') and isinstance(st.session_state.get('diff_html_output'), str) and st.session_state.get('diff_html_output', "").strip().lower().startswith("<p style='color:red;'>error:"):
     st.markdown("---") # Add separator before error
     st.error("Failed to generate visual comparison:")
     st.markdown(st.session_state.diff_html_output, unsafe_allow_html=True)
elif st.session_state.get('processing_comparison'):
     # Show spinner if the processing flag is set
     st.markdown('<div class="loader-container"><div class="loader"></div><p>Processing...</p></div>', unsafe_allow_html=True)

