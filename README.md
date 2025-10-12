# Neuroblastoma — GAN-Augmented & Differentially Private ML

Reproducible pipeline for the paper project **“GAN‑Augmented and Differentially Private ML for Biomedical Systems: A Case Study in Neuroblastoma.”**  
This repo provides a clean, modular implementation to:
- Load the neuroblastoma EHR dataset
- Perform leakage‑safe preprocessing and splits
- Train baselines (Logistic Regression / Random Forest / MLP)
- Train a simple **tabular GAN** to synthesize additional samples
- Re‑train models with **GAN‑augmented** data
- Train a DP classifier with **Opacus** (DP‑SGD) for privacy
- Evaluate AUROC, AUPRC, **ECE**, **Brier**, and **calibration** (isotonic), exporting figures and CSV summaries

> **Dataset (public):**  
> GSE3960_cleaned_data.csv — https://davidechicco.github.io/neuroblastoma_EHRs_data/datasets/GSE3960_cleaned_data.csv

---

## Quickstart

```bash
# 1) Python 3.10+ is recommended
python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Verify GPU for torch
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 4) Run the end-to-end pipeline (uses ./configs/experiment.yaml)
python scripts/run_pipeline.py --config configs/experiment.yaml
```

Outputs are written to `runs/`:
- `runs/figures/` — ROC, PR, calibration curves (PNG)
- `runs/metrics/metrics.json` — key metrics (per stage)
- `runs/tables/` — CSV summaries
- `runs/synth/` — optional synthetic samples (CSV)

You can also open the original Colab notebook under `notebooks/` to compare.

---

## Project layout

```
neuroblastoma-gan-dp-ml/
├── configs/
│   └── experiment.yaml
├── notebooks/
│   └── GAN_ML_improvement_v3.ipynb
├── scripts/
│   └── run_pipeline.py
├── src/neuro_gan_dp/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── augment.py
│   ├── privacy.py
│   ├── evaluate.py
│   └── utils.py
├── runs/                 # created at runtime
├── figures/              # (optional) precomputed figures copied here
├── requirements.txt
├── Makefile
├── LICENSE
└── README.md
```

---

## Configuration

Edit [`configs/experiment.yaml`](configs/experiment.yaml) as needed:
- `data.url`: CSV path/URL (default: GSE3960_cleaned_data.csv URL)
- `data.target_col`: name of the label column (set this correctly!)
- `data.positive_label`: positive class value (for AUROC/AUPRC)
- `preprocess.drop_cols`: identifiers/leaky columns to drop
- `augment.enabled`: enable/disable GAN generation
- `privacy.enabled`: enable/disable differentially private training

If you’re unsure of the actual column names, run:
```bash
python -c "import pandas as pd; import sys; df=pd.read_csv('https://davidechicco.github.io/neuroblastoma_EHRs_data/datasets/GSE3960_cleaned_data.csv'); print(df.head()); print(df.columns.tolist())"
```

> **Important**: This scaffold is leakage‑safe by design (split before scaling, calibrate on validation, apply on test).  
> Adjust `target_col` and any `drop_cols` to match the dataset schema.

---

## Reproducing the paper figures

- Figures are auto‑exported to `runs/figures/` when the pipeline finishes.
- If you already have figures, drop them into `figures/` (we copied any you uploaded).

---

## Citing / Acknowledgements

- Dataset courtesy of the Neuroblastoma EHRs data project by Davide Chicco et al.
- DP via **Opacus**; tabular GAN is a simple MLP‑GAN intended as a lightweight, reproducible baseline.
- Calibration uses scikit‑learn isotonic regression.

