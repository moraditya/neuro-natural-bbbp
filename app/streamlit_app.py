import sys
import os

# Add project root so we can import src.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.inference import predict_smiles

st.set_page_config(page_title="Neuro-Natural BBBP", page_icon="🧠")

st.title("Neuro-Natural: BBB Permeability Predictor")

st.markdown(
    """
Paste a SMILES string for a small molecule and this app will:

- compute RDKit descriptors (MW, logP, HBD, TPSA, QED)
- run both a Random Forest (descriptors) and a GNN (molecular graph)
- return the predicted probability of crossing the blood–brain barrier (BBBP)
"""
)

smiles = st.text_input("Enter SMILES", value="CCOC(=O)C1=CC=CC=C1")

if st.button("Predict") and smiles.strip():
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES string.")
        else:
            st.subheader("Molecule")
            st.image(Draw.MolToImage(mol), caption=smiles)

            rf_prob, gnn_prob, desc = predict_smiles(smiles)

            st.subheader("Predicted BBB Permeability")
            st.write(f"**Random Forest (descriptors):** {rf_prob:.3f}")

            if gnn_prob is None:
                st.write(
                    "**GNN (graph):** not applicable "
                    "(molecule has no bonds, so a graph representation can't be built)."
                )
            else:
                st.write(f"**GNN (graph):** {gnn_prob:.3f}")

            st.subheader("Descriptors")
            st.json(desc)

    except Exception as e:
        st.error(f"Error: {e}")
