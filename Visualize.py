import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

THRESHOLD = 0.5


# ----------------------------------------------------------------------
# Figure 4 -- ROC over all held-out subjects
# ----------------------------------------------------------------------
def plot_roc(df, out_path):
    y, s = df["Target"].to_numpy(), df["Pred"].to_numpy()
    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"EEGTransformer (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance")
    plt.xlim(0, 1); plt.ylim(0, 1.02)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"Subject-level ROC (pooled across folds, n = {len(df)})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figure 4 -> {out_path}   (pooled AUC = {roc_auc:.3f})")
    return roc_auc


# ----------------------------------------------------------------------
# Figure 5 -- per-subject risk score
# ----------------------------------------------------------------------
def plot_risk_scores(df, out_path, threshold=THRESHOLD):
    df = df.sort_values(["Target", "Pred"]).reset_index(drop=True)
    idx = np.arange(len(df))
    ctrl = df["Target"] == 0
    pat = df["Target"] == 1

    plt.figure(figsize=(9, 6))
    plt.scatter(idx[ctrl], df.loc[ctrl, "Pred"], c="blue", marker="o",
                s=55, label="Healthy subject")
    plt.scatter(idx[pat], df.loc[pat, "Pred"], c="red", marker="x",
                s=70, linewidths=2, label="Schizophrenia")
    plt.axhline(threshold, color="gray", linestyle="--", label="Decision boundary")

    plt.xlabel("Subject index (cross-validation)")
    plt.ylabel("Continuous risk score (exploratory)")
    plt.title(f"Subject-level risk score distribution (n = {len(df)})")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figure 5 -> {out_path}")


# ----------------------------------------------------------------------
# Figure 3 -- gradient saliency, one value per electrode
# ----------------------------------------------------------------------
def compute_saliency(model, x, device):
    """
    x: tensor (batch, time, channels)
    returns: ndarray (channels,)  -- mean |d logit / d input| over batch and time
    """
    import torch

    model.eval()
    x = x.to(device).clone().requires_grad_(True)

    logits = model(x)
    # sum to a scalar so a batch larger than 1 can be backpropagated in one pass;
    # gradients w.r.t. each sample's input are unaffected by the sum
    logits.sum().backward()

    # (batch, time, channels) -> average over batch (0) and time (1) -> (channels,)
    return x.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()


def plot_saliency(saliency, out_path):
    plt.figure(figsize=(11, 4.5))
    plt.bar(np.arange(len(saliency)), saliency, color="steelblue")
    plt.xlabel("EEG channel index")
    plt.ylabel("Mean |gradient|")
    plt.title("Gradient-based spatial saliency (per electrode)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    order = np.argsort(saliency)[::-1][:10]
    print(f"Figure 3 -> {out_path}")
    print("  top 10 channels by saliency:",
          ", ".join(f"Ch{i+1}({saliency[i]:.3g})" for i in order))


def run_saliency(checkpoint, data_path, out_path, n_windows, device_str):
    import torch
    from train import EEGTransformer          # same directory

    device = torch.device(device_str if torch.cuda.is_available()
                          or device_str == "cpu" else "cpu")

    model = EEGTransformer().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    X = np.load(data_path, mmap_mode="r")[:n_windows].astype(np.float32)
    # same instance normalization as train.py
    for i in range(X.shape[0]):
        mu = X[i].mean(axis=0, keepdims=True)
        sd = X[i].std(axis=0, keepdims=True)
        X[i] = (X[i] - mu) / (sd + 1e-4)

    sal = compute_saliency(model, torch.tensor(X), device)
    np.save(os.path.splitext(out_path)[0] + "_weights.npy", sal)
    plot_saliency(sal, out_path)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-csv",
                    help="subject_predictions.csv written by evaluate.py (Figs 4 and 5)")
    ap.add_argument("--checkpoint", help="model .pth for saliency (Fig 3)")
    ap.add_argument("--data", default="master_dataset_X_clean_unified.npy",
                    help="preprocessed windows, used with --checkpoint")
    ap.add_argument("--n-windows", type=int, default=8,
                    help="windows to average saliency over")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.subject_csv:
        df = pd.read_csv(args.subject_csv)
        if df["PatientID"].duplicated().any():
            raise SystemExit("subject_predictions.csv has duplicate PatientID rows; "
                             "it should contain one row per subject.")
        plot_roc(df, os.path.join(args.outdir, "Figure_4_ROC.png"))
        plot_risk_scores(df, os.path.join(args.outdir, "Figure_5_RiskScores.png"))

    if args.checkpoint:
        run_saliency(args.checkpoint, args.data,
                     os.path.join(args.outdir, "Figure_3_Saliency.png"),
                     args.n_windows, args.device)

    if not args.subject_csv and not args.checkpoint:
        ap.error("give --subject-csv (Figs 4, 5) and/or --checkpoint (Fig 3)")


if __name__ == "__main__":
    main()
