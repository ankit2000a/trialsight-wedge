import streamlit as st
import io
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from base64 import b64encode
import cairosvg

def convert_svg_to_png(svg_string):
    """Converts an SVG string to a PNG byte string."""
    return cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))

st.set_page_config(page_title="Structure Comparator", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stFileUploader > div > div > button { display: none; } /* Hide the default button */
    .stButton>button { width: 100%; }
    .css-1offfwp p { font-size: 1.1rem; } /* Make 'Drag and drop' text larger */
</style>
""", unsafe_allow_html=True)

st.title("🔬 ChemCompare")
st.markdown("Visually compare two molecules by uploading their `.mol` files and see the differences highlighted.")

# --- Session State Initialization ---
if 'file1' not in st.session_state:
    st.session_state['file1'] = None
if 'file2' not in st.session_state:
    st.session_state['file2'] = None
if 'diff_image' not in st.session_state:
    st.session_state['diff_image'] = None

# --- File Uploader and Display Logic ---
col1, col2 = st.columns(2)

with col1:
    uploaded_file1 = st.file_uploader("Upload Molecule 1 (.mol)", type=['mol'], key="f1")
    if uploaded_file1:
        st.session_state['file1'] = uploaded_file1.getvalue().decode("utf-8")
        st.info(f"Loaded: `{uploaded_file1.name}`")

with col2:
    uploaded_file2 = st.file_uploader("Upload Molecule 2 (.mol)", type=['mol'], key="f2")
    if uploaded_file2:
        st.session_state['file2'] = uploaded_file2.getvalue().decode("utf-8")
        st.info(f"Loaded: `{uploaded_file2.name}`")

# --- Comparison and Display Logic ---
if st.button("Compare Structures", disabled=(not st.session_state.file1 or not st.session_state.file2)):
    with st.spinner("Generating atom-by-atom comparison..."):
        try:
            mol1 = Chem.MolFromMolBlock(st.session_state.file1)
            mol2 = Chem.MolFromMolBlock(st.session_state.file2)

            if mol1 is None or mol2 is None:
                st.error("Error parsing one of the MOL files. Please ensure they are valid.")
            else:
                # Align the molecules
                from rdkit.Chem import AllChem
                AllChem.Compute2DCoords(mol1)
                AllChem.GenerateDepictionMatching2DStructure(mol2, mol1)

                # Get atom and bond differences
                m1_atoms = {a.GetIdx() for a in mol1.GetAtoms()}
                m2_atoms = {a.GetIdx() for a in mol2.GetAtoms()}
                
                m1_bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol1.GetBonds()}
                m2_bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol2.GetBonds()}

                atoms_only_in_1 = list(m1_atoms - m2_atoms)
                atoms_only_in_2 = list(m2_atoms - m1_atoms)
                
                bonds_only_in_1 = [list(b) for b in m1_bonds - m2_bonds]
                bonds_only_in_2 = [list(b) for b in m2_bonds - m1_bonds]
                
                # Highlight the differences
                d = rdMolDraw2D.MolDraw2DSVG(400, 400)
                rdMolDraw2D.PrepareAndDrawMolecule(d, mol1, highlightAtoms=atoms_only_in_1, highlightBonds=bonds_only_in_1)
                d.FinishDrawing()
                svg1 = d.GetDrawingText().replace('svg:','')
                
                d = rdMolDraw2D.MolDraw2DSVG(400, 400)
                rdMolDraw2D.PrepareAndDrawMolecule(d, mol2, highlightAtoms=atoms_only_in_2, highlightBonds=bonds_only_in_2)
                d.FinishDrawing()
                svg2 = d.GetDrawingText().replace('svg:','')

                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Molecule 1")
                    st.image(svg1)
                with col_b:
                    st.subheader("Molecule 2")
                    st.image(svg2)
                    
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")