import os
import numpy as np
import pandas as pd
import pywt
import mne
from glob import glob
from tqdm import tqdm

# Silence MNE warnings globally for a clean terminal
mne.set_log_level('ERROR')

# ==========================================
# 1. Hyperparameters & Configuration
# ==========================================
WINDOW_MINUTES = 3
MAX_WINDOWS = 48        # Cap to ensure class balance per patient
RESCUE_THRESHOLD = 38   # Minimum windows required to trigger rescue/padding (~80%)
WAVELET_TYPE = 'db3'    # Sharper transient preservation
WAVELET_LEVEL = 3       # Protects Theta, thresholds Gamma
EOG_PROXY = 'Ch1'       # Virtual EOG channel (Update to 'E1' if your BIDS channels are named E1, E2, etc.)

DATA_DIR = './ds004000'
PARTICIPANTS_FILE = os.path.join(DATA_DIR, 'participants.tsv')

# ==========================================
# 2. Helper Functions
# ==========================================
def load_labels():
    """Parses the BIDS participants.tsv to map subject IDs to binary labels."""
    if not os.path.exists(PARTICIPANTS_FILE):
        raise FileNotFoundError(f"Could not find {PARTICIPANTS_FILE}.")
    df = pd.read_csv(PARTICIPANTS_FILE, sep='\t')
    label_map = {}
    target_column = next((col for col in ['group', 'diagnosis', 'condition', 'status', 'phenotype'] if col in df.columns), None)

    for index, row in df.iterrows():
        raw_id = str(row['participant_id'])
        numeric_id = ''.join(filter(str.isdigit, raw_id))
        if not numeric_id: continue
        formatted_id = f"sub-{int(numeric_id):03d}"

        diagnosis = str(row[target_column]).lower()
        if any(kw in diagnosis for kw in ['control', 'healthy', 'hc']):
            label_map[formatted_id] = 0.0
        elif any(kw in diagnosis for kw in ['schizo', 'p', 'patient']):
            label_map[formatted_id] = 1.0
            
    return label_map

def denoise_eeg_channel(channel_data, wavelet=WAVELET_TYPE, level=WAVELET_LEVEL):
    """Applies non-linear soft thresholding to scrub localized high-frequency noise."""
    coeffs = pywt.wavedec(channel_data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(channel_data)))
    coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:])
    return pywt.waverec(coeffs, wavelet)[:len(channel_data)]

def extract_and_rescue_windows(patient_data, patient_label, patient_id_numeric, sfreq):
    """Slices the continuous array into windows and rescues patients who are slightly short."""
    window_length = int(sfreq * 60 * WINDOW_MINUTES)
    step_size = int(sfreq * 15) # 15-second sliding step
    
    windows_X, labels_y, groups = [], [], []
    total_steps = patient_data.shape[1]

    # Extract all available windows
    for start in range(0, total_steps - window_length, step_size):
        end = start + window_length
        window = patient_data[:, start:end]
        if window.shape[1] == window_length:
            windows_X.append(window.T) # Transpose to (Time, Channels)
            labels_y.append(patient_label)
            groups.append(patient_id_numeric)

        if len(windows_X) == MAX_WINDOWS: break

    # THE RESCUE LOGIC: Oversample if they are short but above the threshold
    actual_windows = len(windows_X)
    if actual_windows > 0 and actual_windows < MAX_WINDOWS:
        if actual_windows >= RESCUE_THRESHOLD:
            print(f"  -> Rescuing sub-{patient_id_numeric:03d}: Padding from {actual_windows} up to {MAX_WINDOWS} windows.")
            while len(windows_X) < MAX_WINDOWS:
                idx_to_copy = np.random.randint(0, actual_windows)
                windows_X.append(windows_X[idx_to_copy])
                labels_y.append(patient_label)
                groups.append(patient_id_numeric)
        else:
            print(f"  -> WARNING: Dropping sub-{patient_id_numeric:03d}. Too little data ({actual_windows}/{MAX_WINDOWS}).")
            return [], [], []

    return windows_X, labels_y, groups

# ==========================================
# 3. Main End-to-End Pipeline
# ==========================================
if __name__ == "__main__":
    print("Starting Unified BIDS Extraction & Cleaning Pipeline...")
    label_map = load_labels()
    master_X, master_y, master_groups = [], [], []
    
    subject_folders = sorted(glob(os.path.join(DATA_DIR, 'sub-*')))

    for folder in tqdm(subject_folders, desc="Processing Patients"):
        subject_id = os.path.basename(folder)
        if subject_id not in label_map: continue
        label = label_map[subject_id]
        numeric_id = int(''.join(filter(str.isdigit, subject_id)))
        
        vhdr_files = sorted(glob(os.path.join(folder, 'eeg', '*.vhdr')))
        if not vhdr_files: continue

        # ------------------------------------------
        # STEP A: Load & Harmonize Raw Data
        # ------------------------------------------
        raw_runs = []
        for vhdr_file in vhdr_files:
            try:
                raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose='ERROR')
                if len(raw.ch_names) > 128:
                    raw.pick_channels(raw.ch_names[:128])
                if len(raw.ch_names) == 128:
                    raw_runs.append(raw)
            except Exception:
                pass

        if not raw_runs: continue
        
        # Harmonize calibration metadata to bypass MNE's strict concatenation checks
        if len(raw_runs) > 1:
            base_cals = raw_runs[0]._cals
            for r in raw_runs[1:]:
                r._cals = base_cals

        combined_raw = mne.concatenate_raws(raw_runs)
        
        # Ensure our virtual EOG proxy name matches the actual channel names in the file
        actual_eog_proxy = EOG_PROXY if EOG_PROXY in combined_raw.ch_names else combined_raw.ch_names[0]

        # ------------------------------------------
        # STEP B: Mathematical Preprocessing
        # ------------------------------------------
        # 1. FIR Filter
        combined_raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)

        # 2. Downsample to 250 Hz (Prevents the 346 GiB Memory Error)
        combined_raw.resample(250.0)
        true_sfreq = 250.0

        # 3. FastICA
        ica = mne.preprocessing.ICA(n_components=15, random_state=42, method='fastica', verbose=False)
        ica.fit(combined_raw, verbose=False)
        
        eog_indices, _ = ica.find_bads_eog(combined_raw, ch_name=actual_eog_proxy, verbose=False)
        muscle_indices, _ = ica.find_bads_muscle(combined_raw, verbose=False)
        ica.exclude = list(set(eog_indices + muscle_indices))
        
        clean_raw = ica.apply(combined_raw.copy(), verbose=False)
        clean_data = clean_raw.get_data()

        # 4. Wavelet Denoising
        for ch in range(clean_data.shape[0]):
            clean_data[ch, :] = denoise_eeg_channel(clean_data[ch, :])

        # ------------------------------------------
        # STEP C: Windowing & Rescue
        # ------------------------------------------
        X_wins, y_labels, group_ids = extract_and_rescue_windows(clean_data, label, numeric_id, true_sfreq)
        
        if X_wins:
            master_X.extend(X_wins)
            master_y.extend(y_labels)
            master_groups.extend(group_ids)

    # ------------------------------------------
    # Final Save
    # ------------------------------------------
    X_array = np.array(master_X, dtype=np.float32)
    y_array = np.array(master_y, dtype=np.float32)
    groups_array = np.array(master_groups, dtype=np.int32)

    np.save('master_dataset_X_clean_unified.npy', X_array)
    np.save('master_dataset_y_unified.npy', y_array)
    np.save('master_dataset_groups_unified.npy', groups_array)

    print(f"\n✅ Pipeline Complete! Final Clean Dataset Shape: {X_array.shape}")
