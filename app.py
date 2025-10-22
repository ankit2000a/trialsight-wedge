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
    # Reverted to simpler state keys
    for key in ['files', 'diff_html', 'summary', 'original_text', 'revised_text']:
        if key not in st.session_state:
            st.session_state[key] = None
    if 'processing_comparison' not in st.session_state: # Use specific flag
        st.session_state.processing_comparison = False

init_session_state()

# --- Gemini API Configuration ---
ai_enabled = False
try:
    # Attempt to get API key from environment variables first, then secrets
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key and hasattr(st, 'secrets'): # Check if st.secrets exists
         # Use the secrets management provided by Streamlit
         secrets = st.secrets
         if "GOOGLE_API_KEY" in secrets:
            api_key = secrets["GOOGLE_API_KEY"]


    if api_key:
        genai.configure(api_key=api_key)
        # --- Using 2.5 PRO MODEL ---
        model = genai.GenerativeModel('gemini-2.5-pro')
        ai_enabled = True # Assume enabled if configuration works
    else:
        st.warning("Google API Key not found in environment variables or Streamlit secrets. The AI Summary feature is disabled.")
        ai_enabled = False # Explicitly disable if no key

except ImportError:
    st.warning("Streamlit secrets management not available in this environment. Ensure API key is set as an environment variable if running locally without secrets.")
    ai_enabled = False # Disable if secrets can't be imported
except AttributeError:
    # Fallback if st.secrets doesn't exist (older Streamlit versions?)
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
        text = "\n".join(page.get_text("text", sort=True) for page in doc) # Added sort=True

        # --- SIMPLIFIED CLEANING Steps ---
        # Fix common PDF ligature issues
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')

        # Basic whitespace cleanup: replace multiple spaces/tabs with single space, but keep newlines
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove leading/trailing whitespace from each line
        lines = text.splitlines()
        cleaned_lines = [line.strip() for line in lines]
        text = "\n".join(cleaned_lines)
        # Remove multiple blank lines
        text = re.sub(r'\n\s*\n', '\n', text)

        # Remove leading/trailing whitespace from the whole text block AFTER line processing
        text = text.strip()
        # --- END SIMPLIFIED CLEANING Steps ---

        # Important: Return text preserving internal whitespace and line breaks as much as possible
        return text
    except Exception as e:
        # Provide more context in the error message
        st.error(f"Error reading {filename}: {e}")
        return None # Return None on error


def generate_diff_html(text1, text2, filename1="Original", filename2="Revised"):
    """Creates a side-by-side HTML diff of two texts."""
     # Check if text extraction failed
    if text1 is None or text2 is None:
        return "Error: Cannot generate diff because text extraction failed for one or both files."

    d = difflib.HtmlDiff(wrapcolumn=80)
    # This now gets a proper list of lines
    html = d.make_table(text1.splitlines(), text2.splitlines(), fromdesc=filename1, todesc=filename2)
    style = """
    <style>
    table.diff { font-family: monospace; border-collapse: collapse; width: 100%; }
    .diff_header { background-color: #374151; color: #E5E7EB; }
    .diff_add { background-color: rgba(16, 185, 129, 0.2); color: #6EE7B7; text-decoration: none; } /* Green for additions */
    .diff_chg { background-color: rgba(209, 163, 23, 0.2); color: #FCD34D; } /* Yellow for changes */
    .diff_sub { background-color: rgba(239, 68, 68, 0.2); color: #FCA5A5; text-decoration: line-through; } /* Red for deletions */
    </style>
    """
    return style + html

def get_ai_summary(text1, text2):
    """Generates a summary categorizing added/deleted lines using the Gemini API."""
    if not ai_enabled:
        return "AI Summary feature is not available."

     # Check if text extraction failed before diffing
    if text1 is None or text2 is None:
        return "AI Summary cannot be generated because text extraction failed."

    # Generate the diff based on the minimally cleaned text
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    # Use context=0 to only get changed lines
    diff = list(difflib.unified_diff(lines1, lines2, fromfile='Original', tofile='Revised', n=0))

    # Filter for actual change lines (+ or -), ignoring file headers
    # Also ignore lines that are purely whitespace changes after the +/-
    # --- This is the variable we should use ---
    diff_text_lines = [line for line in diff if line.startswith(('+', '-')) and not line.startswith(('---', '+++')) and line[1:].strip()]

    # Check if *any* non-whitespace difference was found
    changes_found = bool(diff_text_lines) # Use the correct variable

    if not changes_found:
         # Check if the original diff had *any* lines (even ignored whitespace ones)
         original_diff_lines = [line for line in diff if line.startswith(('+', '-')) and not line.startswith(('---', '+++'))]
         if original_diff_lines:
             return "No substantive textual differences were found. Detected differences relate only to minor formatting (e.g., line breaks, spacing)."
         else:
            return "No textual differences were found between the documents after minimal cleaning."


    # Join the *meaningful* lines for the prompt
    # --- FIX THE TYPO HERE ---
    diff_text = "".join(diff_text_lines) # Use the correct variable name

    # --- REFINED PROMPT ---
    prompt = f"""
    You are a document comparison assistant focusing on changes in a revised clinical trial protocol. Analyze the ADDED (+) and DELETED (-) lines provided below, which represent the difference between an original and a revised document.

    **Instructions:**
    1.  **Identify Clinically Significant Additions/Deletions:** Review the ADDED (+) and DELETED (-) lines. Identify ONLY those lines that represent additions or deletions related to these key clinical trial areas:
        * Inclusion/Exclusion criteria
        * Dosage information or treatment schedules
        * Study procedures or assessments
        * Safety reporting requirements
        * Key objectives or endpoints
        Try to determine the section or context (e.g., "Inclusion Criteria section") where the change occurred based on surrounding text in the original document (not explicitly provided here, but infer if possible).
    2.  **Identify Other Additions/Deletions:** Identify ALL OTHER ADDED (+) or DELETED (-) lines provided below that DO NOT fall into the clinically significant category. This includes changes to references, general text, titles, etc.
    3.  **IGNORE:** Do NOT report changes *within* a line (only whole added/deleted lines). Do NOT report the addition/removal of blank lines or lines containing only whitespace.
    4.  **Output Format:** Structure your response EXACTLY as follows:

        **Clinically Significant Changes in Revised Document:**
        * [List ONLY the significant ADDED (+) or DELETED (-) lines here, mentioning the context/location if possible (e.g., "+ Added age requirement (10-50) in Inclusion Criteria"). If none, state "None found."]

        **Other Added/Deleted Lines in Revised Document:**
        * [List ALL OTHER non-blank ADDED (+) or DELETED (-) lines here. If none, state "None found."]

    **Detected Added (+) and Deleted (-) Lines (excluding blank lines):**
    ---
    {diff_text[:8000]}
    ---

    Begin Summary:
    """


    try:
        # Basic API call with safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        # Keep temperature slightly lower for consistency
        generation_config = genai.types.GenerationConfig(temperature=0.2)

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )


        if not response.candidates:
             # More detailed feedback extraction
             block_reason = "Unknown"
             safety_ratings_str = "N/A"
             if hasattr(response, 'prompt_feedback'):
                 feedback = response.prompt_feedback
                 if hasattr(feedback, 'block_reason'):
                    block_reason = feedback.block_reason
                 if hasattr(feedback, 'safety_ratings'):
                     safety_ratings_str = ", ".join([f"{rating.category}: {rating.probability}" for rating in feedback.safety_ratings])

             return (f"Error: AI response blocked. Reason: {block_reason}. "
                     f"Safety Ratings: [{safety_ratings_str}]. The prompt or content might violate safety policies.")


        if response.text:
            return response.text.strip()
        else:
             # Check if blocked due to safety even if candidates exist but text is empty
            finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
            if finish_reason == 'SAFETY':
                 safety_ratings_str = ", ".join([f"{rating.category}: {rating.probability}" for rating in response.candidates[0].safety_ratings])
                 return f"Error: AI model returned an empty response, potentially due to safety filters. Finish Reason: {finish_reason}. Safety Ratings: [{safety_ratings_str}]"
            else:
                 return f"Error: AI model returned an empty response. Finish Reason: {finish_reason}"


    except Exception as e:
        # More specific error handling if possible
        error_message = f"Error communicating with the AI model: {e}"
        # You could check for specific error types, e.g., related to quotas
        if "quota" in str(e).lower():
             error_message += "\nThis might be a quota issue. Please check your API key usage and limits."
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
            # Clear previous results when new files are uploaded
            st.session_state.diff_html = None
            st.session_state.summary = None
            st.session_state.original_text = None
            st.session_state.revised_text = None
            st.session_state.processing_comparison = False # Ensure flag is reset
            st.rerun()
else:
    col1, col2 = st.columns(2)
    # Ensure file data exists before accessing name attribute
    file1_name = st.session_state.file1_data.name if st.session_state.get('file1_data') else "N/A"
    file2_name = st.session_state.file2_data.name if st.session_state.get('file2_data') else "N/A"
    with col1:
        st.success(f"Original: **{file1_name}**")
    with col2:
        st.success(f"Revised: **{file2_name}**")

    if st.button("Clear Files and Start Over"):
        # Clear specific keys, not all session state
        keys_to_clear = ['file1_data', 'file2_data', 'diff_html', 'summary',
                         'original_text', 'revised_text', 'processing_comparison']
        for key in keys_to_clear:
             if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# --- Main Logic ---
# Trigger comparison only when the button is clicked and files are present
if st.session_state.get('file1_data') and st.session_state.get('file2_data'):
    # Only show compare button if results aren't already displayed AND not currently processing
    if not st.session_state.get('diff_html') and not st.session_state.get('processing_comparison', False):
        if st.button("Compare Documents", type="primary", use_container_width=True):
            st.session_state.processing_comparison = True # Set flag
            # Clear previous results before processing
            st.session_state.diff_html = None
            st.session_state.summary = None
            st.session_state.original_text = None
            st.session_state.revised_text = None
            st.rerun() # Rerun to show spinner and process

    # --- Processing logic runs if flag is set ---
    if st.session_state.get('processing_comparison', False):
         with st.spinner("Reading, cleaning, and comparing documents..."):
            file1 = st.session_state.file1_data
            file2 = st.session_state.file2_data

            # Check if files are still available (might be cleared)
            if file1 and file2:
                file1_bytes = file1.getvalue()
                file2_bytes = file2.getvalue()

                # Extract single version of text with minimal cleaning
                text1 = extract_text_from_bytes(file1_bytes, file1.name)
                text2 = extract_text_from_bytes(file2_bytes, file2.name)

                # Store text in session state only if extraction was successful
                if text1 is not None and text2 is not None:
                    st.session_state['original_text'] = text1
                    st.session_state['revised_text'] = text2
                    # Generate diff only if text extraction succeeded
                    st.session_state['diff_html'] = generate_diff_html(text1, text2, file1.name, file2.name)
                else:
                    # Error is handled in extract_text_from_bytes, ensure state is clean
                    st.error("Text extraction failed for one or both files. Cannot compare.") # Added user message
                    st.session_state['diff_html'] = None
                    st.session_state['summary'] = None
                    st.session_state['original_text'] = None # Clear text state on error
                    st.session_state['revised_text'] = None

            else:
                 st.error("File data missing, cannot process comparison.") # Handle missing file data case
                 st.session_state['diff_html'] = None
                 st.session_state['summary'] = None


            st.session_state.processing_comparison = False # Reset flag after processing
            # Rerun only if extraction and diff generation didn't fail
            if st.session_state.get('diff_html') and "Error:" not in st.session_state.diff_html:
                 st.rerun() # Rerun to display results


# --- Display Results Section ---
# Display results only if processing is complete and diff_html exists (and not empty error string)
if not st.session_state.get('processing_comparison', False) and st.session_state.get('diff_html') and "Error:" not in st.session_state.get('diff_html', ""):

    # --- DEBUGGER (using the single text version) ---
    with st.expander("Show/Hide Extracted Text (For Debugging)"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Text from Original (Cleaned)")
            # Ensure text exists before displaying
            st.text_area("Original Text", st.session_state.get('original_text', 'N/A'), height=200, key="debug_text1")
        with col2:
            st.subheader("Text from Revised (Cleaned)")
            st.text_area("Revised Text", st.session_state.get('revised_text', 'N/A'), height=200, key="debug_text2")

    # --- Side-by-Side Diff ---
    with st.expander("Show/Hide Side-by-Side Diff", expanded=True):
        st.components.v1.html(st.session_state.diff_html, height=600, scrolling=True)

    st.markdown("---")

    # --- AI Summary (Now requires button click again) ---
    st.subheader("🤖 AI-Powered Summary")
    # Updated description based on new prompt
    st.markdown("Click the button below for a categorized summary of added/deleted lines.")

    # Disable button if AI isn't enabled OR if text extraction failed
    button_disabled = not ai_enabled or st.session_state.get('original_text') is None or st.session_state.get('revised_text') is None

    # --- Updated button text ---
    if st.button("✨ Get Categorized Line Summary", use_container_width=True, disabled=button_disabled, key="generate_summary"):
        with st.spinner("Analyzing and categorizing added/deleted lines..."):
            summary = get_ai_summary(st.session_state.original_text, st.session_state.revised_text)
            st.session_state['summary'] = summary
            st.rerun() # Rerun to display the summary immediately

    # Display the summary if it exists in the session state
    if st.session_state.get('summary'):
         # --- Updated Summary Title ---
         st.markdown("### Categorized Summary of Added/Deleted Lines:")
         # Display summary, handling potential errors returned from get_ai_summary
         summary_text = st.session_state.summary
         if "Error:" in summary_text:
             st.error(summary_text)
         else:
             # Use markdown with code block for better readability of +/- lines
             st.markdown(f"```markdown\n{summary_text}\n```")


    # Explain why button might be disabled more clearly
    elif button_disabled:
         if not ai_enabled:
             st.warning("AI Summary button disabled: Google API Key not configured.")
         elif st.session_state.get('original_text') is None or st.session_state.get('revised_text') is None:
             st.warning("AI Summary button disabled: Text could not be extracted successfully from one or both files.")


# Display loading indicator if processing is ongoing (outside the main results display block)
elif st.session_state.get('processing_comparison', False):
     st.markdown('<div class="loader-container"><div class="loader"></div><p>Processing...</p></div>', unsafe_allow_html=True)

# Display error message if diff generation itself failed
elif st.session_state.get('diff_html') and "Error:" in st.session_state.get('diff_html', ""):
    st.error(st.session_state.diff_html)

