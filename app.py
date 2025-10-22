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
    # Adding keys for the different text versions
    for key in ['files', 'diff_html', 'summary', 
                'original_text_display', 'revised_text_display', 
                'original_text_ai', 'revised_text_ai']:
        if key not in st.session_state:
            st.session_state[key] = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

init_session_state()

# --- Gemini API Configuration ---
ai_enabled = False
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro') # Using the model you found
        ai_enabled = True
    else:
        st.warning("Google API Key not found. The AI Summary feature is disabled.")

except Exception as e:
    ai_enabled = False
    st.warning(f"Could not initialize Google AI: {e}")


# --- Helper Functions ---

@st.cache_data
def extract_text_from_bytes(file_bytes, filename="file"):
    """Extracts and CLEANS text from PDF bytes, returning two versions."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        # --- Version 1: For Display (keeps line breaks) ---
        text_display = "\n".join(page.get_text() for page in doc)
        # Fix ligatures
        text_display = re.sub(r'ﬁ', 'fi', text_display)
        text_display = re.sub(r'ﬂ', 'fl', text_display)
        # Clean up extra spaces on each line but keep line structure
        lines_display = text_display.splitlines()
        cleaned_lines_display = [" ".join(line.split()) for line in lines_display] # Handles multiple spaces between words
        text_display = "\n".join(cleaned_lines_display).strip()

        # --- Version 2: For AI Comparison (aggressive cleaning, ignores line breaks) ---
        text_ai = text_display # Start with the display version
        text_ai = text_ai.lower() # Convert to lowercase
        text_ai = re.sub(r'([:.,;()])', r' \1 ', text_ai) # Add space around punctuation
        text_ai = re.sub(r'\s+', ' ', text_ai) # Collapse ALL whitespace (incl. newlines) to single space
        text_ai = text_ai.strip()

        return text_display, text_ai
        
    except Exception as e:
        st.error(f"Error reading {filename}: {e}")
        return None, None # Return None if reading fails

def generate_diff_html(text1_display, text2_display, filename1="Original", filename2="Revised"):
    """Creates a side-by-side HTML diff using the display text."""
    # Only compare if text is not None
    if text1_display is None or text2_display is None:
        return "Error: Could not generate diff due to text extraction issues."
        
    d = difflib.HtmlDiff(wrapcolumn=80)
    # Use the display text (with line breaks) for the HTML table
    html = d.make_table(text1_display.splitlines(), text2_display.splitlines(), fromdesc=filename1, todesc=filename2)
    style = """
    <style>
    table.diff { font-family: monospace; border-collapse: collapse; width: 100%; }
    .diff_header { background-color: #374151; color: #E5E7EB; }
    .diff_add { background-color: #052e16; } /* Green for additions */
    .diff_chg { background-color: #4d380c; } /* Yellow for changes */
    .diff_sub { background-color: #4c1d1d; } /* Red for deletions */
    </style>
    """
    return style + html

def get_ai_summary(text1_ai, text2_ai):
    """Generates a summary of changes using the cleaned AI text."""
    if not ai_enabled:
        return "AI Summary feature is not available."
    
    # Only compare if text is not None
    if text1_ai is None or text2_ai is None:
        return "AI Summary cannot be generated due to text extraction issues."

    # --- Use the aggressively cleaned text (text_ai) for diffing ---
    diff = list(difflib.unified_diff(
        text1_ai.splitlines(keepends=True), # Split even if it's one line
        text2_ai.splitlines(keepends=True), # Split even if it's one line
        fromfile='Original_AI',
        tofile='Revised_AI',
    ))
    diff_text = "".join([line for line in diff if line.startswith(('+', '-')) and not line.startswith(('---', '+++'))])

    # Check if the *only* differences are whitespace changes ignored by the AI version
    diff_display_only = list(difflib.unified_diff(
        st.session_state.original_text_display.splitlines(keepends=True),
        st.session_state.revised_text_display.splitlines(keepends=True),
         fromfile='Original_Display',
        tofile='Revised_Display',
    ))
    diff_display_text = "".join([line for line in diff_display_only if line.startswith(('+', '-')) and not line.startswith(('---', '+++'))])


    if not diff_text.strip():
        if diff_display_text.strip(): # Differences exist in display but not AI version
             return ("No substantive textual differences were found after cleaning and normalization. "
                     "Minor differences in formatting, line breaks, or spacing were detected but ignored.")
        else: # No differences found at all
            return "No textual differences were found between the documents."


    # --- MODIFIED PROMPT ---
    prompt = f"""
    You are an expert clinical trial protocol reviewer. Analyze the following changes between two versions of a document (differences in formatting, case, and extra whitespace have been minimized) and generate a concise, bulleted list of the most significant modifications.
    Focus specifically on substantive changes related to:
    1.  Inclusion/Exclusion criteria
    2.  Dosage information or treatment schedules
    3.  Study procedures or assessments
    4.  Safety reporting requirements
    5.  Key objectives or endpoints

    Ignore minor grammatical corrections or rephrasing unless it alters the clinical meaning.
    If you find significant changes, list them.
    If the changes you found seem minor or non-substantive AFTER ignoring formatting/whitespace, explicitly state that.

    Here are the extracted textual changes (lines starting with '+' were added, lines with '-' were removed):
    ---
    {diff_text[:8000]} 
    ---
    **Summary of Key Changes:**
    """
    
    try:
        # Add retry logic with backoff
        max_retries = 3
        delay = 1
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                # Check if response has text, otherwise retry might be needed
                if response.text:
                   return response.text
                else:
                    # Handle cases where the API returns an empty response unexpectedly
                    if attempt == max_retries - 1:
                       return "Error: AI model returned an empty response."
                    # print(f"Empty response received, retrying in {delay}s...") # Console logging removed
            except Exception as e:
                # Handle potential API errors (like rate limiting)
                if attempt == max_retries - 1:
                    return f"Error communicating with the AI model after multiple retries: {e}"
                # print(f"API Error: {e}. Retrying in {delay}s...") # Console logging removed
            
            time.sleep(delay)
            delay *= 2 # Exponential backoff
        
        return "Error: AI model failed to generate summary after multiple retries." # Fallback message

    except Exception as e:
        # General catch-all for unexpected errors during the process
        return f"An unexpected error occurred while generating the AI summary: {e}"


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
            # Store uploaded files
            st.session_state.file1_data = uploaded_files[0]
            st.session_state.file2_data = uploaded_files[1]
            # --- AUTO-TRIGGER COMPARISON ---
            st.session_state.processing_comparison = True # Set flag to trigger comparison logic
            st.session_state.diff_html = None
            st.session_state.summary = None
            st.rerun() # Rerun to execute the comparison logic immediately
else:
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"Original: **{st.session_state.file1_data.name}**")
    with col2:
        st.success(f"Revised: **{st.session_state.file2_data.name}**")

    if st.button("Clear Files and Start Over"):
        # Clear all relevant session state keys
        keys_to_clear = ['file1_data', 'file2_data', 'diff_html', 'summary', 
                         'original_text_display', 'revised_text_display', 
                         'original_text_ai', 'revised_text_ai', 'processing_comparison']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# --- Main Logic ---
# --- MOVED: Triggered automatically after file upload or if files exist ---
if st.session_state.get('file1_data') and st.session_state.get('file2_data') and st.session_state.processing_comparison:
    # Check if text has already been processed in this run to prevent infinite loop
    if 'original_text_display' not in st.session_state or st.session_state.original_text_display is None:
        
        with st.spinner("Reading, cleaning, and comparing documents..."):
            file1 = st.session_state.file1_data
            file2 = st.session_state.file2_data
            
            file1_bytes = file1.getvalue()
            file2_bytes = file2.getvalue()
            
            # Extract both versions of text
            text1_display, text1_ai = extract_text_from_bytes(file1_bytes, file1.name)
            text2_display, text2_ai = extract_text_from_bytes(file2_bytes, file2.name)
            
            # Store both versions in session state
            st.session_state['original_text_display'] = text1_display
            st.session_state['revised_text_display'] = text2_display
            st.session_state['original_text_ai'] = text1_ai
            st.session_state['revised_text_ai'] = text2_ai

            # Proceed only if text extraction was successful
            if text1_display is not None and text2_display is not None:
                # Generate HTML diff using display text
                st.session_state['diff_html'] = generate_diff_html(text1_display, text2_display, file1.name, file2.name)
                
                # --- AUTO-GENERATE SUMMARY ---
                with st.spinner("Analyzing changes with Gemini AI..."):
                     st.session_state['summary'] = get_ai_summary(text1_ai, text2_ai) # Use AI text

            else:
                st.error("Failed to process one or both PDF files. Cannot compare or summarize.")
                st.session_state['diff_html'] = None # Ensure diff is cleared on error
                st.session_state['summary'] = None # Ensure summary is cleared on error
            
            st.session_state.processing_comparison = False # Reset flag
            st.rerun() # Rerun to update the display now that processing is done

# --- Display Results Section ---
# Display results only if processing is complete and diff_html exists
if not st.session_state.get('processing_comparison', False) and st.session_state.get('diff_html'):
    
    # --- DEBUGGER ---
    with st.expander("Show/Hide Extracted Text (For Debugging)"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Text for Display (Original)")
            st.text_area("Display Text 1", st.session_state.original_text_display, height=150, key="debug_text1_display")
            st.subheader("Text for AI (Original)")
            st.text_area("AI Text 1", st.session_state.original_text_ai, height=150, key="debug_text1_ai")
        with col2:
            st.subheader("Text for Display (Revised)")
            st.text_area("Display Text 2", st.session_state.revised_text_display, height=150, key="debug_text2_display")
            st.subheader("Text for AI (Revised)")
            st.text_area("AI Text 2", st.session_state.revised_text_ai, height=150, key="debug_text2_ai")
            
    # --- Side-by-Side Diff ---
    with st.expander("Show/Hide Side-by-Side Diff", expanded=True):
        st.components.v1.html(st.session_state.diff_html, height=600, scrolling=True)

    st.markdown("---")
    
    # --- AI Summary (Now shown automatically) ---
    st.subheader("🤖 AI-Powered Summary")
    if st.session_state.get('summary'):
         st.markdown(st.session_state.summary)
    else:
        # Show a placeholder or message if summary isn't ready or failed
        if ai_enabled:
             st.info("Summary is being generated or encountered an issue.")
        else:
             st.warning("AI Summary feature is disabled. No API Key found.")

# Display loading indicator if processing is ongoing
elif st.session_state.get('processing_comparison', False):
     st.markdown('<div class="loader-container"><div class="loader"></div><p>Processing...</p></div>', unsafe_allow_html=True)
