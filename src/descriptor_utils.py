# src/descriptor_utils.py
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, QED

def compute_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    mol_wt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Lipinski.NumHDonors(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    qed = QED.qed(mol)

    return {
        "smiles": smiles,
        "mol_wt": mol_wt,
        "logp": logp,
        "h_donors": h_donors,
        "tpsa": tpsa,
        "qed": qed,
    }
