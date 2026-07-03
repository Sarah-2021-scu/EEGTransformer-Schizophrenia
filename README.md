# VRAM-Optimized EEGTransformer for Schizophrenia Classification

This repository contains the official PyTorch implementation and preprocessing pipeline for our manuscript submitted to *ARRAY*: **"Transformer-Based Modeling of Electroencephalography for Schizophrenia Classification and Risk Score."**

## Overview
We propose a VRAM-optimized Transformer framework utilizing Flash Attention to process continuous 3-minute EEG windows (128 channels × 45,000 timepoints). The model performs binary classification (Schizophrenia vs. Healthy Control) and generates a Continuous Clinical Risk Score.

## Dataset
This pipeline is designed to run on the publicly available NEMAR ds004000 dataset.
1. Download the dataset from the [NEMAR Repository](https://nemar.org/dataexplorer/detail?dataset_id=ds004000).
2. **Important:** Ensure you download the data in the **BrainVision format** (`.vhdr`, `.vmrk`, `.eeg`), as the preprocessing script relies on the `read_raw_brainvision` function.
3. Update the `DATA_DIR` path in `prepare_dataset.py` to point to your local extraction folder.

## Environment Setup
To replicate the conda/pip environment used in this study:
```bash
pip install -r requirements.txt

Reproducibility
1. Preprocessing:
Run the hybrid ICA-wavelet pipeline to clean the raw BIDS data, apply the db3 Level 3 thresholding, and extract the 3-minute windows into .npy tensors.

Bash
python prepare_dataset.py
2. Training & Evaluation:
Execute the group-stratified 5-fold cross-validation. This script utilizes PyTorch's sdpa_kernel(SDPBackend.FLASH_ATTENTION) and requires a CUDA-enabled GPU (developed on NVIDIA RTX 6000 Ada).

Bash
python train.py
