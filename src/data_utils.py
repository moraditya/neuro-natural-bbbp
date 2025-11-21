from pathlib import Path
from typing import Iterable

import deepchem as dc
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED


def _compute_properties(smiles: str) -> pd.Series:
    """Return basic physicochemical properties for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None

    if not mol:
        return pd.Series(
            {
                "mol_wt": pd.NA,
                "logp": pd.NA,
                "h_donors": pd.NA,
                "tpsa": pd.NA,
                "qed": pd.NA,
            }
        )

    return pd.Series(
        {
            "mol_wt": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "h_donors": Descriptors.NumHDonors(mol),
            "tpsa": Descriptors.TPSA(mol),
            "qed": QED.qed(mol),
        }
    )


def prepare_neuro_natural_data(
    output_path: str | Path = "data/processed/neuro_natural_dataset.csv",
    splitter: str = "scaffold",
    source_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load the BBBP dataset from a local CSV (if provided) or via DeepChem,
    compute simple descriptors, flag natural-product-like molecules, and save to CSV.

    Returns the processed DataFrame.
    """
    if source_csv is None:
        local_csv = Path("data/raw/BBBP.csv")
        source_csv = local_csv if local_csv.exists() else None

    if source_csv:
        source_csv = Path(source_csv)
        df_raw = pd.read_csv(source_csv)
        # DeepChem's BBBP uses 'smiles' and target column 'p_np' (1 = permeable)
        target_col = "p_np" if "p_np" in df_raw.columns else df_raw.columns[-1]
        df = df_raw.rename(columns={target_col: "permeable"})[["smiles", "permeable"]]
    else:
        # Falls back to downloading via DeepChem (requires internet)
        _, datasets, _ = dc.molnet.load_bbbp(splitter=splitter)
        train_dataset, _, _ = datasets
        df = pd.DataFrame(train_dataset.ids, columns=["smiles"])
        df["permeable"] = train_dataset.y

    props = df["smiles"].apply(_compute_properties)
    df = pd.concat([df, props], axis=1).dropna()
    df["is_natural_like"] = (df["mol_wt"] > 500) & (df["h_donors"] > 5)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


__all__: Iterable[str] = ["prepare_neuro_natural_data"]


if __name__ == "__main__":
    # Running the module directly will generate the processed dataset.
    out = prepare_neuro_natural_data()
    print(f"Saved {len(out)} rows to {Path('data/processed/neuro_natural_dataset.csv').resolve()}")
