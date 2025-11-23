# Neuro-Natural: Predicting Blood–Brain Barrier Permeability with Graph Neural Networks

**Live demo:**  
https://huggingface.co/spaces/moraditya/Neuro-Natural-BloodBrainBarrier-Predictor

This repository contains an end-to-end pipeline for predicting whether small molecules can cross the blood–brain barrier (BBB) using:

- Cheminformatics descriptors computed with RDKit
- A Random Forest baseline model
- A Graph Neural Network (GNN) implemented with PyTorch Geometric
- A Streamlit web application, containerized with Docker and deployed on HuggingFace Spaces

The project also includes exploratory data analysis (EDA), misclassification analysis, and a modular codebase suitable for further extension.

---

## 1. Overview

The blood–brain barrier is a highly selective interface that restricts the passage of most circulating molecules into the central nervous system. For CNS drug discovery, an early and critical question is:

> Given a candidate molecule, how likely is it to cross the blood–brain barrier?

This project addresses that question by:

1. Computing standard physicochemical descriptors and drug-likeness scores.
2. Training a descriptor-based Random Forest classifier as a baseline.
3. Training a graph-based GNN that operates directly on molecular structures.
4. Deploying an interactive screening tool, allowing users to input SMILES strings and obtain predictions.

The resulting system is designed to be clear, reproducible, and easily extendable.

---

## 2. Live Demo

The model is exposed via an interactive Streamlit app on HuggingFace Spaces:

**HuggingFace Space:**  
https://huggingface.co/spaces/moraditya/Neuro-Natural-BloodBrainBarrier-Predictor

Functionality:

- Input: SMILES string of a small molecule
- Output:
  - Rendering of the molecule
  - RDKit descriptors: molecular weight (MW), logP, H-bond donors, TPSA, QED
  - Random Forest BBB permeability probability
  - GNN BBB permeability probability (when a valid graph can be formed)

For single-atom molecules or other cases where no bonds are present, the app returns descriptor-based predictions and marks the GNN prediction as not applicable.

---

## 3. Dataset

The project uses the **BBBP (Blood–Brain Barrier Penetration)** dataset from **MoleculeNet**, a benchmark collection for molecular property prediction.

- Task: Binary classification — BBB-permeable vs non-permeable
- Label: `permeable`
  - `1` = crosses BBB
  - `0` = does not cross BBB
- Input: SMILES strings representing small molecules

After preprocessing, the label distribution on the processed dataset is approximately:

- `permeable = 1.0`: ~82%
- `permeable = 0.0`: ~18%

This is a class-imbalanced problem, and evaluation metrics such as AUC and F1-score are used in addition to raw accuracy.

---

## 4. Feature Engineering (Cheminformatics)

All descriptors are computed using **RDKit**. For each molecule, the following are calculated:

- Lipinski-style features:
  - `mol_wt`: Molecular weight
  - `logp`: Octanol-water partition coefficient (MolLogP)
  - `h_donors`: Number of hydrogen bond donors (NumHDonors)
  - (H-bond acceptors are also computed during EDA, but not necessarily used as model features)
- Polar surface:
  - `tpsa`: Topological Polar Surface Area (TPSA)
- Drug-likeness:
  - `qed`: Quantitative Estimate of Drug-likeness (QED)

These descriptors serve several purposes:

- Exploratory data analysis and visualization of BBB permeability vs physicochemical properties.
- Input features for the Random Forest baseline.
- Descriptor-based prediction pathway in the deployed app (always available, regardless of graph feasibility).

---

## 5. Exploratory Data Analysis (EDA)

EDA is primarily conducted in `notebooks/01_data_descriptors.ipynb`. Key observations include:

- **Class imbalance:** Most molecules are labeled as permeable, which influences model selection and evaluation. AUC and F1-score are considered more informative than accuracy alone.
- **Permeability trends:**
  - Permeable molecules generally have:
    - Lower TPSA
    - Higher logP
    - Higher QED scores
  - Non-permeable molecules tend to:
    - Have higher TPSA (more polar)
    - Contain more hydrogen bond donors

Kernel density estimates and distribution plots confirm that molecules crossing the BBB often occupy a region of chemical space characterized by moderate molecular weight, higher lipophilicity, and limited polarity, although there is significant overlap between classes.

These trends motivate the use of both descriptor-based models and structure-aware GNNs.

---

## 6. Models

### 6.1 Random Forest Baseline

Implemented in: `notebooks/02_baseline_rf.ipynb`

**Features used:**

```text
["mol_wt", "logp", "h_donors", "tpsa", "qed"]

Model: `RandomForestClassifier` from scikit-learn  
Trained on the descriptor set with a train/test split  
Hyperparameters tuned using cross-validation  
Class imbalance accounted for in evaluation  

Representative performance (on held-out test set):

| Metric     | Score |
|------------|-------|
| AUC        | ~0.88 |
| F1-score   | ~0.92 |
| Precision  | ~0.90 |
| Recall     | ~0.95 |
| Accuracy   | ~0.87 |

The Random Forest serves as a strong baseline using only global descriptors and no explicit structural information.

## 6.2 Graph Neural Network (GNN)

Implemented in: `notebooks/03_gnn_bbbp.ipynb` and `src/gnn_model.py`

### Graph Representation
- **Nodes:** atoms  
- **Edges:** bonds between atoms  
- **Node features** (from `src/featurization.py`), such as:  
  - Atom type (one-hot)  
  - Atom degree  
  - Formal charge  
  - Aromaticity  
  - Hybridization state  

Graphs are represented using **PyTorch Geometric** (`torch_geometric.data.Data`).

### Architecture
- **Model:** `BBBP_GCN`  
- Several GCN layers with ReLU activations  
- Global pooling (mean/sum) to obtain a graph-level embedding  
- Final linear classifier predicting permeable / non-permeable  

Representative performance (test set):

| Metric    | Score      |
|-----------|------------|
| AUC       | ~0.84–0.85 |
| F1-score  | ~0.88      |

The Random Forest slightly outperforms the GNN on this dataset, but the GNN uses structural information and provides useful inductive bias.

## 7. Misclassification Analysis

### False positives  
(predicted permeable, actually non-permeable)
- Moderate–high logP  
- Moderate TPSA  
- Higher QED than typical non-permeable molecules  
- Chemically resemble plausible BBB-penetrant structures

### False negatives  
(predicted non-permeable, actually permeable)
- Higher TPSA  
- Lower logP  
- Chemically violate typical BBB-permeable heuristics

This shows limits of descriptor/structure-only models. True BBB permeability depends on transporters, efflux, metabolism, 3D conformation, etc.


## 8. Repo Structure
neuro-natural-bbbp/
├─ data/
│  ├─ raw/ # Original BBBP / MoleculeNet data (not tracked)
│  └─ processed/ # Processed CSVs with descriptors + labels
│
├─ notebooks/
│  ├─ 01_data_descriptors.ipynb
│  ├─ 02_baseline_rf.ipynb
│  └─ 03_gnn_bbbp.ipynb
│
├─ src/
│  ├─ __init__.py
│  ├─ data_utils.py
│  ├─ featurization.py
│  ├─ gnn_model.py
│  ├─ descriptor_utils.py
│  └─ inference.py
│
├─ models/
│  ├─ rf_bbbp.joblib
│  └─ gnn_bbbp.pt
│
├─ app/
│  └─ streamlit_app.py
│
├─ Dockerfile
├─ README.md
└─ requirements.txt


## 9. Installation and Local Usage

### 9.1 Clone repo
```bash
git clone https://github.com/moraditya/neuro-natural-bbbp.git
cd neuro-natural-bbbp

### 9.2 Install dependencies
conda create -n neuro-natural-bbbp python=3.10
conda activate neuro-natural-bbbp

pip install -r requirements.txt

## 10. Running Notebooks
Run in order:

01_data_descriptors.ipynb

02_baseline_rf.ipynb

03_gnn_bbbp.ipynb

Pretrained models are included in models/

## 11. Running the App

paste streamlit run app/streamlit_app.py in your ClI to run locally

## 12. Docker & HuggingFace Deployment

### Local Docker
```bash
docker build -t neuro-natural-bbbp .
docker run -p 7860:7860 neuro-natural-bbbp
Runs at:
http://localhost:7860

Deploy to HuggingFace Spaces
Create a Space (SDK: Docker)

Add remote:

bash
Copy code
git remote add hf https://huggingface.co/spaces/<user>/<space-name>
Push:

bash
Copy code
git push hf main
Already deployed at:
https://huggingface.co/spaces/moraditya/Neuro-Natural-BloodBrainBarrier-Predictor

## 13. Limitations & Future Work
Limitations:

Uses only 2D SMILES + graphs (no 3D)

Doesn't model biological effects (transporters, efflux, metabolism)

Trained only on BBBP dataset

Future extensions:

Add 3D descriptors / 3D GNNs

Multi-task ADMET prediction

Stronger GNNs (GIN, MPNNs)

Probability calibration (Platt, isotonic)

## 14. References
Zhenqin Wu et al., MoleculeNet, Chemical Science, 2018

RDKit — https://www.rdkit.org

PyTorch Geometric — https://pytorch-geometric.readthedocs.io/

## 15. License
MIT License (see LICENSE)



