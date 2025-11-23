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
