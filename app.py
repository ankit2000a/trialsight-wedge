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
    td { padding: 0.1em 0.3em; vertical-align: top; white-space: pre-wrap; } /* Ensure wrapping */

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
    # Store both display and AI versions of text
    for key in ['file1_data', 'file2_data', 'diff_html', 'summary',
                'original_text_display', 'revised_text_display',
                'original_text_ai', 'revised_text_ai']:
        if key not in st.session_state:
            st.session_state[key] = None
    if 'processing_comparison' not in st.session_state:
        st.session_state.processing_comparison = False

init_session_state()

# --- Gemini API Configuration ---
ai_enabled = False
api_key = None
try:
    api_key_env = os.environ.get("GOOGLE_API_KEY")
    api_key_secret = None
    # Check if Streamlit's secrets management is available and use it
    if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
         api_key_secret = st.secrets["GOOGLE_API_KEY"]

    # Prioritize secrets, then environment variable
    if api_key_secret:
        api_key = api_key_secret
    elif api_key_env:
        api_key = api_key_env

    # Configure API if key was found
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        ai_enabled = True
    else:
        st.warning("Google API Key not found in Streamlit secrets or environment variables. AI Summary disabled.")
        ai_enabled = False

except ImportError:
    # Handle case where st.secrets might not be available (older Streamlit?)
    if api_key_env:
        api_key = api_key_env
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        ai_enabled = True
        st.info("Using API key from environment variable (secrets management not available).")
    else:
        st.warning("Streamlit secrets management not available and GOOGLE_API_KEY environment variable not set. AI Summary disabled.")
        ai_enabled = False
except Exception as e:
    ai_enabled = False
    st.warning(f"Could not initialize Google AI: {e}")

# --- Helper Functions ---

@st.cache_data
def extract_text_from_bytes(file_bytes, filename="file"):
    """Extracts text, returning a display version and an AI comparison version."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_display = "\n".join(page.get_text("text", sort=True) for page in doc)

        # --- Minimal Cleaning for Display ---
        text_display = text_display.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text_display = re.sub(r'[ \t]+', ' ', text_display)
        lines_display = text_display.splitlines()
        cleaned_lines_display = [line.strip() for line in lines_display]
        text_display = "\n".join(cleaned_lines_display)
        text_display = re.sub(r'\n\s*\n+', '\n\n', text_display) # Keep paragraph breaks
        text_display = text_display.strip()

        # --- Aggressive Cleaning for AI Comparison ---
        text_ai = text_display # Start with display text
        text_ai = text_ai.lower() # Lowercase
        # Remove punctuation might be too aggressive, let's keep it simpler
        # text_ai = re.sub(r'[^\w\s-]', '', text_ai)
        text_ai = re.sub(r'\s+', ' ', text_ai) # Collapse ALL whitespace to single space
        text_ai = text_ai.strip()

        return text_display, text_ai
    except Exception as e:
        st.error(f"Error reading {filename}: {e}")
        return None, None


def generate_diff_html(text1_display, text2_display, filename1="Original", filename2="Revised"):
    """Creates side-by-side HTML diff using the display text version."""
    if text1_display is None or text2_display is None:
        return "Error: Cannot generate diff because text extraction failed."

    d = difflib.HtmlDiff(wrapcolumn=80, tabsize=4)
    html = d.make_table(text1_display.splitlines(), text2_display.splitlines(), fromdesc=filename1, todesc=filename2)
    style = f"<style>{difflib.HtmlDiff._styles}</style>" # Use built-in styles
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

def get_ai_summary(text1_ai, text2_ai):
    """Generates a summary categorizing ADDED/DELETED lines using the AI text version."""
    if not ai_enabled: return "AI Summary feature is not available."
    if text1_ai is None or text2_ai is None: return "AI Summary cannot be generated: text extraction failed."

    # Compare the aggressively cleaned, single-line versions
    lines1_ai = text1_ai.splitlines(keepends=True)
    lines2_ai = text2_ai.splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines1_ai, lines2_ai, fromfile='Original_AI', tofile='Revised_AI', n=0))

    # Filter for non-blank added/deleted lines based on the AI text diff
    diff_lines = [line for line in diff if line.startswith(('+', '-')) and not line.startswith(('---', '+++')) and line[1:].strip()]

    if not diff_lines:
         # Check original diff (display text) for context if AI diff is empty
        # Ensure display texts exist before attempting diff
         if 'original_text_display' not in st.session_state or 'revised_text_display' not in st.session_state or \
           st.session_state.original_text_display is None or st.session_state.revised_text_display is None:
             # Handle case where display text might not be available yet
            return "Cannot determine context as original display text is missing."

         lines1_disp = st.session_state.original_text_display.splitlines(keepends=True)
         lines2_disp = st.session_state.revised_text_display.splitlines(keepends=True)
         original_diff_all = list(difflib.unified_diff(lines1_disp, lines2_disp, fromfile='Original', tofile='Revised', n=0))
         original_diff_lines_all = [line for line in original_diff_all if line.startswith(('+', '-')) and not line.startswith(('---', '+++'))]

         if any(line[1:].strip() for line in original_diff_lines_all):
             return "No substantive textual differences found after aggressive cleaning (ignoring case and detailed whitespace variations). Original differences appeared minor."
         elif original_diff_lines_all:
             return "No substantive textual differences found. Only whitespace or blank line changes detected."
         else:
            return "No textual differences were found between the documents after cleaning."

    diff_text_for_prompt = "".join(diff_lines)

    # --- FINAL PROMPT v4 --- (Revised for clarity)
    prompt = f"""
    Analyze the differences between two versions of a clinical trial protocol based *only* on the provided ADDED (+) and DELETED (-) lines, which represent non-blank line changes after aggressive cleaning (lowercase, collapsed whitespace). Categorize these line changes.

    **Instructions:**
    1.  **Clinically Significant Lines:** Identify ADDED (+) or DELETED (-) lines clearly related to:
        * Inclusion/Exclusion criteria
        * Dosage / Treatment schedules
        * Procedures / Assessments
        * Safety reporting
        * Objectives / Endpoints
        Infer the context/section if possible (e.g., "Inclusion Criteria"). List these first.
    2.  **Other Added Lines:** List ALL OTHER provided ADDED (+) lines not in the significant category.
    3.  **Other Deleted Lines:** List ALL OTHER provided DELETED (-) lines not in the significant category.
    4.  **IGNORE:** Do not mention blank lines or simple whitespace changes as they are excluded from the input list. Do not analyze changes *within* lines.
    5.  **Output Format:** Structure your response EXACTLY like this:

        **Clinically Significant Changes (Added/Deleted Lines):**
        * [List ONLY significant ADDED (+) or DELETED (-) lines here, mentioning context. If none found, state "None found."]

        **Other Added Lines:**
        * [List ALL OTHER non-blank ADDED (+) lines from the input here. If none found, state "None found."]

        **Other Deleted Lines:**
        * [List ALL OTHER non-blank DELETED (-) lines from the input here. If none found, state "None found."]

    **Detected Added (+) and Deleted (-) Non-Blank Lines (Aggressively Cleaned):**
    ---
    {diff_text_for_prompt[:8000]}
    ---

    Begin Summary:
    """


    try:
        # Relax safety settings slightly, BLOCK_NONE might be too permissive for some content
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_LOW_AND_ABOVE"}, # Was BLOCK_NONE
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_LOW_AND_ABOVE"}, # Was BLOCK_NONE
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_LOW_AND_ABOVE"}, # Was BLOCK_NONE
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_LOW_AND_ABOVE"}, # Was BLOCK_NONE
        ]
        generation_config = genai.types.GenerationConfig(
            temperature=0.1, # Keep low for consistency
            # max_output_tokens=2048 # Increase if summaries get cut off
            )

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Enhanced Response Handling
        if not response.candidates:
            block_reason = "Unknown"
            safety_ratings_str = "N/A"
            if hasattr(response, 'prompt_feedback'):
                feedback = response.prompt_feedback
                if hasattr(feedback, 'block_reason'): block_reason = feedback.block_reason
                if hasattr(feedback, 'safety_ratings'): safety_ratings_str = ", ".join([f"{rating.category.name}: {rating.probability.name}" for rating in feedback.safety_ratings]) # Use names
            return f"Error: AI response blocked. Reason: {block_reason}. Safety Ratings: [{safety_ratings_str}]."

        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else "Unknown"
        safety_ratings_str = "N/A"
        if hasattr(candidate, 'safety_ratings'):
             safety_ratings_str = ", ".join([f"{rating.category.name}: {rating.probability.name}" for rating in candidate.safety_ratings]) # Use names


        # Check for content and expected finish reason
        if candidate.content and candidate.content.parts and candidate.content.parts[0].text:
             response_text = candidate.content.parts[0].text.strip()
             if finish_reason not in ['STOP', 'MAX_TOKENS']:
                 st.warning(f"AI response finished unexpectedly ({finish_reason}). Safety: [{safety_ratings_str}]. Result might be incomplete.")
             return response_text
        else: # Handle empty text response explicitly
             return f"Error: AI model returned an empty response. Finish Reason: {finish_reason}. Safety Ratings: [{safety_ratings_str}]"


    except Exception as e:
        error_message = f"Error communicating with the AI model: {e}"
        st.error(error_message) # Show error in UI
        # Check for quota error in exception message
        if "quota" in str(e).lower() or "429" in str(e):
             st.error("Quota issue likely. Check API key usage/limits or wait and retry.")
        return "Summary generation failed due to an error." # Return a user-friendly message

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
            # Clear previous results immediately on new upload
            keys_to_clear = ['diff_html', 'summary', 'original_text_display', 'revised_text_display', 'original_text_ai', 'revised_text_ai']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            st.session_state.processing_comparison = False # Reset flag needed here too
            st.rerun()
else:
    # Display file names and clear button
    col1, col2 = st.columns(2)
    file1_name = st.session_state.file1_data.name if st.session_state.get('file1_data') else "N/A"
    file2_name = st.session_state.file2_data.name if st.session_state.get('file2_data') else "N/A"
    with col1: st.success(f"Original: **{file1_name}**")
    with col2: st.success(f"Revised: **{file2_name}**")

    if st.button("Clear Files and Start Over"):
        keys_to_clear = ['file1_data', 'file2_data', 'diff_html', 'summary', 'original_text_display', 'revised_text_display', 'original_text_ai', 'revised_text_ai', 'processing_comparison']
        for key in keys_to_clear:
             if key in st.session_state: del st.session_state[key]
        st.rerun()

# --- Comparison Logic ---
# Trigger comparison if files are loaded but results aren't computed yet
# OR if the compare button is explicitly clicked
should_run_comparison = False
compare_button_clicked = False

if st.session_state.get('file1_data') and st.session_state.get('file2_data'):
    # Show Compare button only if results don't exist yet and not processing
    if not st.session_state.get('diff_html') and not st.session_state.get('processing_comparison'):
         if st.button("Compare Documents", type="primary", use_container_width=True):
            should_run_comparison = True
            compare_button_clicked = True # Track button click specifically if needed later
            st.session_state.processing_comparison = True # Set flag for spinner
            # Clear results before re-comparing on button click
            keys_to_clear = ['diff_html', 'summary', 'original_text_display', 'revised_text_display', 'original_text_ai', 'revised_text_ai']
            for key in keys_to_clear:
                 if key in st.session_state: del st.session_state[key]
            st.rerun() # Rerun immediately for spinner

# Execute comparison if flagged (either by button or auto-trigger logic if implemented)
if st.session_state.get('processing_comparison'):
     with st.spinner("Reading, cleaning, and comparing documents..."):
        file1 = st.session_state.file1_data
        file2 = st.session_state.file2_data
        # Ensure files still exist (might have been cleared)
        if file1 and file2:
            file1_bytes = file1.getvalue()
            file2_bytes = file2.getvalue()

            # Get both text versions
            text1_display, text1_ai = extract_text_from_bytes(file1_bytes, file1.name)
            text2_display, text2_ai = extract_text_from_bytes(file2_bytes, file2.name)

            if text1_display is not None and text2_display is not None:
                # Store both versions
                st.session_state['original_text_display'] = text1_display
                st.session_state['revised_text_display'] = text2_display
                st.session_state['original_text_ai'] = text1_ai
                st.session_state['revised_text_ai'] = text2_ai
                # Generate HTML diff using display version
                st.session_state['diff_html'] = generate_diff_html(text1_display, text2_display, file1.name, file2.name)
            else:
                st.error("Text extraction failed. Cannot compare.") # Error already shown by func, but good to set state
                st.session_state['diff_html'] = None
                st.session_state['summary'] = None
        else:
             st.error("File data missing, cannot process comparison.")
             st.session_state['diff_html'] = None
             st.session_state['summary'] = None

        st.session_state.processing_comparison = False # Reset flag after processing
        # Rerun only if successful to show results
        if st.session_state.get('diff_html') and "Error:" not in st.session_state.diff_html:
             st.rerun()
        # If error occurred, don't rerun, let the error message display

# --- Display Results Section ---
# Only display if comparison is done and diff_html is valid
if not st.session_state.get('processing_comparison') and st.session_state.get('diff_html') and "Error:" not in st.session_state.get('diff_html', ""):

    # --- DEBUGGER ---
    with st.expander("Show/Hide Extracted Text (For Debugging)"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Text for Display (Original)")
            st.text_area("Display Text 1", st.session_state.get('original_text_display', 'N/A'), height=150, key="debug_text1_display")
            st.subheader("Text for AI Compare (Original)")
            st.text_area("AI Text 1", st.session_state.get('original_text_ai', 'N/A'), height=150, key="debug_text1_ai")
        with col2:
            st.subheader("Text for Display (Revised)")
            st.text_area("Display Text 2", st.session_state.get('revised_text_display', 'N/A'), height=150, key="debug_text2_display")
            st.subheader("Text for AI Compare (Revised)")
            st.text_area("AI Text 2", st.session_state.get('revised_text_ai', 'N/A'), height=150, key="debug_text2_ai")

    # --- Side-by-Side Diff (using display text) ---
    with st.expander("Show/Hide Side-by-Side Diff", expanded=True):
        st.components.v1.html(st.session_state.diff_html, height=600, scrolling=True)

    st.markdown("---")

    # --- AI Summary ---
    st.subheader("🤖 AI-Powered Summary")
    st.markdown("Click the button below for a categorized summary of added/deleted lines.")

    # Disable button checks
    button_disabled = not ai_enabled or st.session_state.get('original_text_ai') is None or st.session_state.get('revised_text_ai') is None

    if st.button("✨ Get Categorized Line Summary", use_container_width=True, disabled=button_disabled, key="generate_summary"):
        # Ensure AI text versions exist before calling summary
        if st.session_state.get('original_text_ai') is not None and st.session_state.get('revised_text_ai') is not None:
            with st.spinner("Analyzing and categorizing added/deleted lines..."):
                # Use AI text versions for summary generation
                summary = get_ai_summary(st.session_state.original_text_ai, st.session_state.revised_text_ai)
                st.session_state['summary'] = summary
                st.rerun() # Rerun to display summary
        else:
             st.error("Cannot generate summary because cleaned text for AI comparison is missing.")


    # Display Summary Section
    if st.session_state.get('summary'):
         st.markdown("### Categorized Summary of Added/Deleted Lines:")
         summary_text = st.session_state.summary
         # Check for specific error messages returned by get_ai_summary
         if summary_text.startswith("Error:") or "AI Summary cannot be generated" in summary_text or "AI Summary feature is not available" in summary_text:
             st.error(summary_text) # Display as error
         elif "No textual differences" in summary_text or "No substantive textual differences" in summary_text:
              st.info(summary_text) # Display informational message
         else:
             # Display the formatted summary in a code block
             st.markdown(f"```markdown\n{summary_text}\n```")

    # Explain disabled button state
    elif button_disabled and not st.session_state.get('processing_comparison'): # Don't show if still processing
         if not ai_enabled: st.warning("AI Summary disabled: API Key not configured.")
         elif st.session_state.get('original_text_ai') is None or st.session_state.get('revised_text_ai') is None:
             st.warning("AI Summary disabled: Text extraction may have failed.")

# Handle specific errors or loading state outside the main results block
elif st.session_state.get('diff_html') and "Error:" in st.session_state.get('diff_html', ""):
    # Display error from generate_diff_html if it occurred
    st.error(st.session_state.diff_html)
elif st.session_state.get('processing_comparison'):
     # Show spinner if the processing flag is set
     st.markdown('<div class="loader-container"><div class="loader"></div><p>Processing...</p></div>', unsafe_allow_html=True)

