import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch

def plot_roc_curve(y_true, y_scores):
    """Generates Figure 4: Aggregate ROC Curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'EEGTransformer (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Aggregate Patient-Level ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('Figure_4_ROC.png')

def plot_risk_scores(y_true, y_scores):
    """Generates Figure 5: Continuous Risk Score Continuum."""
    plt.figure(figsize=(8, 6))
    
    # Separate by true label
    patients = y_scores[y_true == 1]
    controls = y_scores[y_true == 0]
    
    plt.scatter(np.zeros_like(controls), controls, c='blue', marker='o', alpha=0.6, label='Healthy Controls')
    plt.scatter(np.ones_like(patients), patients, c='red', marker='x', alpha=0.6, label='Schizophrenia Patients')
    
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Decision Boundary')
    plt.xticks([0, 1], ['Controls', 'Patients'])
    plt.ylabel('Predicted Continuous Risk Score')
    plt.title('Risk Score Continuum')
    plt.legend()
    plt.savefig('Figure_5_RiskScores.png')

def generate_saliency_map(model, input_tensor):
    """Generates Figure 3: Gradient-based spatial saliency."""
    model.eval()
    input_tensor.requires_grad_()
    
    output = model(input_tensor)
    output.backward()
    
    # Compute absolute gradient across sequence length, averaged over batch
    saliency = input_tensor.grad.abs().mean(dim=(0, 2)).detach().numpy()
    
    # Save the 1D spatial importance array (can be mapped to topoplot via MNE)
    np.save('saliency_weights.npy', saliency)
    print("Saved spatial saliency weights to saliency_weights.npy")
