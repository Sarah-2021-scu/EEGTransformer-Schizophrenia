import numpy as np
import math

def calculate_cohens_d(mean1, sd1, mean2, sd2):
    """Calculates Cohen's d for aggregate performance comparisons."""
    sd_pooled = math.sqrt((sd1**2 + sd2**2) / 2)
    return abs(mean1 - mean2) / sd_pooled

def bootstrap_superiority_test(transformer_aucs, baseline_aucs, n_iterations=2000, alpha=0.0167):
    """
    Performs a patient-level bootstrap test (B=2000) with Bonferroni correction.
    Expects paired arrays of subject-level AUCs.
    """
    assert len(transformer_aucs) == len(baseline_aucs), "Sample sizes must match."
    n_subjects = len(transformer_aucs)
    differences = np.array(transformer_aucs) - np.array(baseline_aucs)
    
    count_null = 0
    for _ in range(n_iterations):
        # Resample with replacement at the subject level
        resample_indices = np.random.choice(n_subjects, size=n_subjects, replace=True)
        resampled_diffs = differences[resample_indices]
        
        if np.mean(resampled_diffs) <= 0:
            count_null += 1
            
    p_value = count_null / n_iterations
    significant = p_value < alpha
    return p_value, significant

if __name__ == "__main__":
    # 1. Cohen's d Calculations (From Table II)
    d_conformer = calculate_cohens_d(0.875, 0.05, 0.820, 0.152)
    d_eegnet = calculate_cohens_d(0.875, 0.05, 0.791, 0.138)
    d_chrononet = calculate_cohens_d(0.875, 0.05, 0.785, 0.161)
    
    print(f"Cohen's d vs EEG-Conformer: {d_conformer:.3f}")
    print(f"Cohen's d vs EEGNetv4: {d_eegnet:.3f}")
    print(f"Cohen's d vs ChronoNet: {d_chrononet:.3f}")
