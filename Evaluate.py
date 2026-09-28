import argparse
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss

THRESHOLD = 0.5
ECE_BINS = 10


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins=ECE_BINS):
    """Equal-width binning ECE. Returns |accuracy - confidence| weighted by bin size."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if not m.any():
            continue
        acc = (y_true[m] == (y_prob[m] >= THRESHOLD)).mean()
        conf = y_prob[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return ece


def classification_metrics(y_true, y_prob, threshold=THRESHOLD):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec and sens and (prec + sens) else np.nan
    acc = (tp + tn) / len(y_true)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:            # single class in this fold
        auc = np.nan

    return {
        "n": len(y_true),
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "AUC": auc,
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "Precision": prec,
        "F1": f1,
        "Brier": brier_score_loss(y_true, y_prob),
        "ECE": expected_calibration_error(y_true, y_prob),
    }


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
def load_fold(path, epoch=None):
    """Read one fold CSV and collapse window rows to one row per subject."""
    df = pd.read_csv(path)

    required = {"Epoch", "PatientID", "Target", "Probability"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")

    chosen = df["Epoch"].max() if epoch is None else epoch
    df = df[df["Epoch"] == chosen]
    if df.empty:
        raise SystemExit(f"{path}: no rows for epoch {chosen}")

    # subject-level score = mean of that subject's window probabilities
    agg = (df.groupby("PatientID")
             .agg(Target=("Target", "first"),
                  Pred=("Probability", "mean"),
                  Windows=("Probability", "size"))
             .reset_index())

    # sanity: a subject must not carry two different labels
    bad = df.groupby("PatientID")["Target"].nunique()
    if (bad > 1).any():
        raise SystemExit(f"{path}: subject(s) with inconsistent Target values")

    return agg, chosen


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="directory containing fold_*_granular_predictions.csv")
    ap.add_argument("--epoch", type=int, default=None,
                    help="epoch to evaluate (default: last epoch in each file)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--out", default=None, help="output directory (default: run_dir)")
    args = ap.parse_args()

    out_dir = args.out or args.run_dir
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob(os.path.join(args.run_dir, "fold_*_granular_predictions.csv")))
    if not files:
        raise SystemExit(f"No fold_*_granular_predictions.csv found in {args.run_dir}")

    per_fold, subject_rows = [], []

    for path in files:
        fold = os.path.basename(path).split("_")[1]
        agg, epoch_used = load_fold(path, args.epoch)

        m = classification_metrics(agg["Target"].to_numpy(),
                                   agg["Pred"].to_numpy(),
                                   args.threshold)
        m = {"Fold": fold, "Epoch": epoch_used, **m}
        per_fold.append(m)

        agg = agg.assign(Fold=fold, Epoch=epoch_used)
        subject_rows.append(agg)

        print(f"fold {fold} (epoch {epoch_used}): n={m['n']:2d}  "
              f"AUC={m['AUC']:.3f}  acc={m['Accuracy']:.3f}  "
              f"sens={m['Sensitivity']:.3f}  spec={m['Specificity']:.3f}  "
              f"F1={m['F1']:.3f}")

    df_fold = pd.DataFrame(per_fold)
    df_subj = pd.concat(subject_rows, ignore_index=True)

    # ---- guard: every subject must be validated exactly once ----
    dup = df_subj["PatientID"].duplicated().sum()
    if dup:
        print(f"\nWARNING: {dup} subject(s) appear in more than one validation fold. "
              "Cross-validation splitting may be leaking.", file=sys.stderr)

    # ---- pooled (all held-out subjects together) ----
    pooled = classification_metrics(df_subj["Target"].to_numpy(),
                                    df_subj["Pred"].to_numpy(),
                                    args.threshold)

    metric_cols = ["AUC", "Accuracy", "Sensitivity", "Specificity",
                   "Precision", "F1", "Brier", "ECE"]
    summary = pd.DataFrame({
        "Metric": metric_cols,
        "Fold_mean": [df_fold[c].mean() for c in metric_cols],
        "Fold_SD": [df_fold[c].std(ddof=1) for c in metric_cols],
        "Pooled": [pooled[c] for c in metric_cols],
    })

    cm = pd.DataFrame(
        [[pooled["TP"], pooled["FN"]], [pooled["FP"], pooled["TN"]]],
        index=["True: Patient", "True: Healthy"],
        columns=["Pred: Patient", "Pred: Healthy"],
    )

    df_fold.to_csv(os.path.join(out_dir, "metrics_per_fold.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    df_subj.to_csv(os.path.join(out_dir, "subject_predictions.csv"), index=False)
    cm.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

    print("\n--- summary (fold mean +/- SD, and pooled) ---")
    for _, r in summary.iterrows():
        print(f"  {r.Metric:<12} {r.Fold_mean:.3f} +/- {r.Fold_SD:.3f}"
              f"   pooled {r.Pooled:.3f}")

    print(f"\n--- pooled confusion matrix (n = {pooled['n']}) ---")
    print(cm.to_string())

    print(f"\nWrote 4 files to {out_dir}")
    print("Note: fold-mean and pooled values differ; state which the manuscript reports.")


if __name__ == "__main__":
    main()
