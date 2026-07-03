import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, roc_auc_score

# ==========================================
# 0. Setup Checkpoint Directory
# ==========================================
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join("saved_runs", f"final_run_{run_timestamp}")
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"📁 Saving to: {SAVE_DIR}\n")

# ==========================================
# 0. Set Seeds for Reproducibility
# ==========================================
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================================
# 1. Dataset & Architecture
# ==========================================
class EEGDataset(Dataset):
    def __init__(self, X, y, groups, is_training=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.groups = groups
        self.is_training = is_training

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_data = self.X[idx].clone()
        if self.is_training:
            # Data Augmentation (Noise & Masking)
            x_data += torch.randn_like(x_data) * 0.05
            mask = torch.rand(128) > 0.10
            x_data[:, ~mask] = 0.0
        return x_data, self.y[idx], self.groups[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EEGTransformer(nn.Module):
    def __init__(self, input_dim=128, d_model=16, nhead=4, num_layers=2):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(0.3)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.3, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.dropout(self.pos_encoder(self.projection(x)))
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x = self.transformer(x)
        return self.fc_out(self.layer_norm(x.mean(dim=1))).squeeze(-1)

# ==========================================
# 2. Main Execution (Hard 50 Epochs & Granular Logging)
# ==========================================
if __name__ == "__main__":
    print("Loading unified BIDS dataset into RAM...")
    X = np.load('master_dataset_X_clean_unified.npy').astype(np.float16)
    y = np.load('master_dataset_y_unified.npy')
    groups = np.load('master_dataset_groups_unified.npy')

    print("Applying Instance Z-score normalization (In-place)...")
    for i in range(X.shape[0]):
        X_mean = np.mean(X[i], axis=0, keepdims=True)
        X_std = np.std(X[i], axis=0, keepdims=True)
        X[i] = (X[i] - X_mean) / (X_std + 1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_patient_aucs = []

    TOTAL_EPOCHS = 50

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"\n{'='*30}\nStarting Fold {fold + 1} / 5\n{'='*30}")

        # NEW: Create the .log file and write the header
        log_file_path = os.path.join(SAVE_DIR, f'fold_{fold+1}_metrics.log')
        with open(log_file_path, "w") as f:
            f.write(f"--- FOLD {fold+1} TRAINING LOG ---\n")

        train_loader = DataLoader(EEGDataset(X[train_idx], y[train_idx], groups[train_idx], is_training=True), batch_size=4, shuffle=True)
        val_loader = DataLoader(EEGDataset(X[val_idx], y[val_idx], groups[val_idx], is_training=False), batch_size=4, shuffle=False)

        model = EEGTransformer().to(device)
        MAX_LR = 1e-4
        optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-4)
        
        # Adjusted OneCycleLR to perfectly map across exactly 50 epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=MAX_LR, epochs=TOTAL_EPOCHS,
            steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos'
        )
        
        criterion, scaler = nn.BCEWithLogitsLoss(), GradScaler()
        best_auc = 0.0

        # Master list to hold every single window prediction for this fold
        fold_granular_logs = []

        for epoch in range(TOTAL_EPOCHS):
            # --- TRAINING PHASE ---
            model.train()
            train_loss = 0.0
            for batch_X, batch_y, _ in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()

                with autocast(device_type=device.type):
                    loss = criterion(model(batch_X), batch_y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item()

            # --- VALIDATION PHASE ---
            model.eval()
            val_preds, val_targets, val_patient_ids = [], [], []
            with torch.no_grad():
                for batch_X, batch_y, batch_group in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    with autocast(device_type=device.type):
                        logits = model(batch_X)
                    val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
                    val_patient_ids.extend(batch_group.numpy())

            # 1. Granular Logging: Save every window prediction for this epoch
            for w_idx, (pid, tgt, pred) in enumerate(zip(val_patient_ids, val_targets, val_preds)):
                fold_granular_logs.append({
                    'Epoch': epoch + 1,
                    'PatientID': int(pid),
                    'WindowID': w_idx,
                    'Target': float(tgt),
                    'Probability': float(pred)
                })

            # 2. Window-Level AUC
            val_auc = roc_auc_score(val_targets, val_preds)

            # 3. Patient-Level Post-Processing
            df = pd.DataFrame({'PatientID': val_patient_ids, 'Target': val_targets, 'Pred': val_preds})
            patient_agg = df.groupby('PatientID').mean().reset_index()
            patient_auc = roc_auc_score(patient_agg['Target'], patient_agg['Pred'])

            epoch_train_loss = train_loss/len(train_loader)
            
            # NEW: Format the log line, print it, and instantly write it to the .log file
            log_line = f"Epoch {epoch+1:02d}/{TOTAL_EPOCHS} | Train Loss: {epoch_train_loss:.6f} | Window AUC: {val_auc:.4f} | **Patient AUC: {patient_auc:.4f}**"
            print(log_line)
            
            with open(log_file_path, "a") as f:
                f.write(log_line + "\n")

            #print(f"Epoch {epoch+1:02d}/{TOTAL_EPOCHS} | Train Loss: {train_loss/len(train_loader):.6f} | Window AUC: {val_auc:.4f} | **Patient AUC: {patient_auc:.4f}**")

            # 4. Checkpoint Logic (No Early Stopping, just saving the best weights)
            if patient_auc > best_auc:
                best_auc = patient_auc
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'best_model_fold_{fold+1}.pth'))
                print(f"   -> ⭐ New Peak Patient AUC! Checkpoint saved.")

        # --- AFTER THE FOLD FINISHES 50 EPOCHS ---
        fold_patient_aucs.append(best_auc)
        
        # Save the granular CSV for this fold
        df_granular = pd.DataFrame(fold_granular_logs)
        csv_path = os.path.join(SAVE_DIR, f'fold_{fold+1}_granular_predictions.csv')
        df_granular.to_csv(csv_path, index=False)
        print(f"💾 Saved {len(df_granular)} window-level logs to {csv_path}")

        # Memory Cleanup
        del model, optimizer, train_loader, val_loader, scheduler, df_granular, fold_granular_logs
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # ==========================================
    # 4. Final Aggregation Logging
    # ==========================================
    print(f"\n✅ Final run complete. All models and granular CSVs saved in: {SAVE_DIR}")
    print(f"Mean Patient AUC Across Folds: {np.mean(fold_patient_aucs):.4f}")
