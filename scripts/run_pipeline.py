from __future__ import annotations
import argparse, os, json
import yaml
import numpy as np

from neuro_gan_dp.utils import set_seed, ensure_dir
from neuro_gan_dp.data import load_dataset, prepare_splits
from neuro_gan_dp.evaluate import evaluate_stage
from neuro_gan_dp.augment import train_gan
from neuro_gan_dp.models import MLPClassifier

# baselines
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier as SKMLP
from sklearn.metrics import roc_auc_score, average_precision_score

import torch, torch.nn as nn

def sigmoid(x): return 1/(1+np.exp(-x))

def fit_sklearn(model, Xtr, ytr, Xval, yval, Xte):
    model.fit(Xtr, ytr)
    p_val = model.predict_proba(Xval)[:,1] if hasattr(model, "predict_proba") else model.decision_function(Xval)
    p_test = model.predict_proba(Xte)[:,1] if hasattr(model, "predict_proba") else model.decision_function(Xte)
    return p_val, p_test

def fit_torch_mlp(Xtr, ytr, Xval, yval, Xte, epochs=50, lr=1e-3, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPClassifier(in_dim=Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xval_t = torch.tensor(Xval, dtype=torch.float32).to(device); yval_t = torch.tensor(yval, dtype=torch.float32).to(device)
    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    best_state, best_val = None, -1e9

    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(); logits = model(xb); loss = loss_fn(logits, yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xval_t)).cpu().numpy()
            auc = roc_auc_score(yval, pv)
            if auc > best_val:
                best_val = auc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_val = torch.sigmoid(model(Xval_t)).cpu().numpy()
        p_test = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32).to(device))).cpu().numpy()
    return p_val, p_test

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("preprocess", {}).get("random_state", 42))

    runs_dir = ensure_dir("runs")
    fig_dir = ensure_dir(os.path.join(runs_dir, "figures"))
    metrics_dir = ensure_dir(os.path.join(runs_dir, "metrics"))
    tables_dir = ensure_dir(os.path.join(runs_dir, "tables"))
    synth_dir = ensure_dir(os.path.join(runs_dir, "synth"))

    # --- Data
    df = load_dataset(cfg["data"]["url"])
    target = cfg["data"]["target_col"]
    drop_cols = cfg["data"].get("drop_cols", [])
    ds = prepare_splits(df, target, drop_cols,
                        test_size=cfg["preprocess"]["test_size"],
                        val_size=cfg["preprocess"]["val_size"],
                        random_state=cfg["preprocess"]["random_state"],
                        scale=cfg["preprocess"]["scale"])
    Xtr, ytr, Xval, yval, Xte, yte = ds.X_train, ds.y_train, ds.X_val, ds.y_val, ds.X_test, ds.y_test

    results = {}

    # --- Baselines
    if cfg["baselines"].get("logistic_regression", True):
        lr = LogisticRegression(max_iter=2000, n_jobs=None)
        pv, pt = fit_sklearn(lr, Xtr, ytr, Xval, yval, Xte)
        results["baseline_lr"] = evaluate_stage(yte, pt, "baseline_lr", fig_dir, y_val=yval, p_val=pv, do_calibration=cfg["calibration"]["enabled"])

    if cfg["baselines"].get("random_forest", True):
        rf = RandomForestClassifier(n_estimators=500, random_state=42)
        pv, pt = fit_sklearn(rf, Xtr, ytr, Xval, yval, Xte)
        results["baseline_rf"] = evaluate_stage(yte, pt, "baseline_rf", fig_dir, y_val=yval, p_val=pv, do_calibration=cfg["calibration"]["enabled"])

    if cfg["baselines"].get("mlp_sklearn", True):
        mlp = SKMLP(hidden_layer_sizes=(128,128), max_iter=400)
        pv, pt = fit_sklearn(mlp, Xtr, ytr, Xval, yval, Xte)
        results["baseline_mlp"] = evaluate_stage(yte, pt, "baseline_mlp", fig_dir, y_val=yval, p_val=pv, do_calibration=cfg["calibration"]["enabled"])

    # --- GAN Augmentation
    if cfg["augment"].get("enabled", True):
        synth = train_gan(
            Xtr,
            n_synth=cfg["augment"]["n_synth"],
            epochs=cfg["augment"]["epochs"],
            batch_size=cfg["augment"]["batch_size"],
            latent_dim=cfg["augment"]["latent_dim"],
            lr=cfg["augment"]["lr"],
        )
        np.savetxt(os.path.join(synth_dir, "synthetic_X.csv"), synth, delimiter=",")
        # naive label reuse via bootstrap (class balance preserved)
        # in practice, consider a conditional GAN; here we mirror y distribution
        pos_ratio = (ytr > 0.5).mean()
        y_synth = (np.random.rand(synth.shape[0]) < pos_ratio).astype(int)
        Xtr_aug = np.vstack([Xtr, synth])
        ytr_aug = np.concatenate([ytr, y_synth])

        lr_aug = LogisticRegression(max_iter=2000)
        pv, pt = fit_sklearn(lr_aug, Xtr_aug, ytr_aug, Xval, yval, Xte)
        results["aug_lr"] = evaluate_stage(yte, pt, "aug_lr", fig_dir, y_val=yval, p_val=pv, do_calibration=cfg["calibration"]["enabled"])

    # --- DP Training (Torch + Opacus)
    if cfg["privacy"].get("enabled", True):
        from neuro_gan_dp.privacy import train_dp_mlp
        dp_model = train_dp_mlp(
            Xtr, ytr, Xval, yval,
            epochs=cfg["privacy"]["epochs"],
            batch_size=cfg["privacy"]["batch_size"],
            lr=cfg["privacy"]["lr"],
            max_grad_norm=cfg["privacy"]["max_grad_norm"],
            noise_multiplier=cfg["privacy"]["noise_multiplier"],
        )
        with torch.no_grad():
            p_val = torch.sigmoid(dp_model(torch.tensor(Xval, dtype=torch.float32))).cpu().numpy()
            p_test = torch.sigmoid(dp_model(torch.tensor(Xte, dtype=torch.float32))).cpu().numpy()
        results["dp_mlp"] = evaluate_stage(yte, p_test, "dp_mlp", fig_dir, y_val=yval, p_val=p_val, do_calibration=cfg["calibration"]["enabled"])

    # Save summary
    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Done. Metrics written to runs/metrics/metrics.json and figures to runs/figures/.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = ap.parse_args()
    main(args.config)
