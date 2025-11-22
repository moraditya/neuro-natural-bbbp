# src/inference.py
import os
import joblib
import torch
import pandas as pd
from rdkit import Chem

from .descriptor_utils import compute_descriptors
from .featurization import smiles_to_data
from .gnn_model import BBBP_GCN

DEVICE = torch.device("cpu")

# Base project dir: .../neuro-natural-bbbp
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RF_PATH = os.path.join(BASE_DIR, "models", "rf_bbbp.joblib")
GNN_PATH = os.path.join(BASE_DIR, "models", "gnn_bbbp.pt")

_MODELS_LOADED = False
_RF_MODEL = None
_GNN_MODEL = None


def load_models():
    """Singleton-style loader to avoid reloading models on every call."""
    global _MODELS_LOADED, _RF_MODEL, _GNN_MODEL

    if _MODELS_LOADED:
        return _RF_MODEL, _GNN_MODEL

    # Load RF
    _RF_MODEL = joblib.load(RF_PATH)

    # Use a molecule with at least one bond so featurization doesn't return None
    dummy = smiles_to_data("CCO", label=0)  # ethanol
    if dummy is None:
        raise RuntimeError("Dummy molecule featurization failed")
    in_channels = dummy.x.size(1)

    _GNN_MODEL = BBBP_GCN(in_channels=in_channels, hidden_dim=64)
    state_dict = torch.load(GNN_PATH, map_location=DEVICE)
    _GNN_MODEL.load_state_dict(state_dict)
    _GNN_MODEL.to(DEVICE)
    _GNN_MODEL.eval()

    _MODELS_LOADED = True
    return _RF_MODEL, _GNN_MODEL


def predict_smiles(smiles: str):
    """
    Run both RF and GNN models on a SMILES string.

    Returns
    -------
    rf_prob : float
        RF probability of being BBB-permeable.
    gnn_prob : float or None
        GNN probability, or None if a graph could not be built
        (e.g., molecule has no bonds).
    desc : dict
        RDKit descriptor dictionary.
    """
    rf_model, gnn_model = load_models()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # RF: descriptors → prob
    desc = compute_descriptors(smiles)
    X = pd.DataFrame([desc])[["mol_wt", "logp", "h_donors", "tpsa", "qed"]]
    rf_prob = float(rf_model.predict_proba(X)[0, 1])

    # GNN: graph → prob (gracefully handle molecules with no bonds)
    data = smiles_to_data(smiles, label=0)
    gnn_prob = None

    if data is not None:
        data = data.to(DEVICE)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            logits = gnn_model(data.x, data.edge_index, batch)
            gnn_prob = torch.softmax(logits, dim=1)[0, 1].item()
    # else: keep gnn_prob = None

    return rf_prob, gnn_prob, desc
