from __future__ import annotations
import numpy as np, json, os
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

def expected_calibration_error(y_true, y_prob, n_bins: int = 10):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        idx = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if idx.sum() == 0: continue
        conf = y_prob[idx].mean()
        acc = (y_true[idx] > 0.5).mean()
        ece += (idx.mean()) * abs(acc - conf)
    return ece

def calibrate_isotonic(y_val, p_val, y_test, p_test):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_val, y_val)
    return iso.transform(p_test)

def plot_curve(xs, ys, xlabel, ylabel, title, path):
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.savefig(path, bbox_inches='tight'); plt.close()

def evaluate_stage(y_true, y_prob, stage_prefix, out_dir, y_val=None, p_val=None, do_calibration=True):
    os.makedirs(out_dir, exist_ok=True)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    ece_pre = expected_calibration_error(y_true, y_prob)
    brier_pre = brier_score_loss(y_true, y_prob)

    # simple ROC/PR via thresholds
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plot_curve(fpr, tpr, "FPR", "TPR", f"ROC — {stage_prefix}", os.path.join(out_dir, f"roc_{stage_prefix}.png"))
    plot_curve(recall, precision, "Recall", "Precision", f"PR — {stage_prefix}", os.path.join(out_dir, f"pr_{stage_prefix}.png"))

    cal_method = "none"
    ece_post, brier_post = None, None
    if do_calibration and y_val is not None and p_val is not None:
        p_test_cal = calibrate_isotonic(y_val, p_val, y_true, y_prob)
        ece_post = expected_calibration_error(y_true, p_test_cal)
        brier_post = brier_score_loss(y_true, p_test_cal)
        # reliability diagram
        # bins
        bins = np.linspace(0,1,11)
        digitized = np.digitize(p_test_cal, bins) - 1
        bin_acc = [ (y_true[digitized==i].mean() if (digitized==i).sum()>0 else np.nan) for i in range(10) ]
        bin_conf = [ (p_test_cal[digitized==i].mean() if (digitized==i).sum()>0 else np.nan) for i in range(10) ]
        plt.figure()
        plt.plot([0,1],[0,1],'--')
        plt.plot(bin_conf, bin_acc, marker='o')
        plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(f"Calibration — {stage_prefix}")
        plt.savefig(os.path.join(out_dir, f"calibration_{stage_prefix}.png"), bbox_inches='tight'); plt.close()
        cal_method = "isotonic"

    metrics = dict(
        AUROC=round(float(auroc),4),
        AUPRC=round(float(auprc),4),
        ECE_pre=round(float(ece_pre),4),
        ECE_post=(None if ece_post is None else round(float(ece_post),4)),
        Brier_pre=round(float(brier_pre),6),
        Brier_post=(None if brier_post is None else round(float(brier_post),6)),
        calibration=cal_method,
    )
    with open(os.path.join(out_dir, f"metrics_{stage_prefix}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
