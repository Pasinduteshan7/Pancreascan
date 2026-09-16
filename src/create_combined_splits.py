# src/create_combined_splits.py
"""
Create combined train/val splits merging MSD Task07 and PANORAMA datasets.

Performs patient-level stratified splitting to ensure:
  1. No patient appears in both train and val
  2. Both MSD and PANORAMA patients are represented in both splits
  3. Tumor-containing ratio is balanced across splits
  4. Split is reproducible (fixed random seed)

Outputs:
  - data/processed/split_combined.json (same format as split.json)
  - data/processed/split_combined_meta.json (metadata about the split)
"""
import os
import sys
import json
import random
import numpy as np
from collections import defaultdict

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MSD_IMAGES_DIR      = os.path.join(PROJECT_DIR, "data", "processed", "images")
MSD_LABELS_DIR      = os.path.join(PROJECT_DIR, "data", "processed", "labels")
PANORAMA_IMAGES_DIR = os.path.join(PROJECT_DIR, "data", "panorama_processed", "images")
PANORAMA_LABELS_DIR = os.path.join(PROJECT_DIR, "data", "panorama_processed", "labels")

OUTPUT_SPLIT   = os.path.join(PROJECT_DIR, "data", "processed", "split_combined.json")
OUTPUT_META    = os.path.join(PROJECT_DIR, "data", "processed", "split_combined_meta.json")

RANDOM_SEED = 42
VAL_RATIO   = 0.20  # 20% validation


def gather_patient_slices(images_dir, labels_dir, dataset_name):
    """
    Scan a processed directory and group slices by patient.
    Returns dict: patient_id -> {slices: [...], has_tumor: bool, dataset: str}
    """
    patients = defaultdict(lambda: {'slices': [], 'has_tumor': False, 'dataset': dataset_name})

    if not os.path.exists(images_dir):
        print(f"WARNING: {images_dir} does not exist, skipping.")
        return patients

    files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])

    for filename in files:
        # Check label exists
        label_path = os.path.join(labels_dir, filename)
        if not os.path.exists(label_path):
            continue

        # Extract patient ID (everything before _slice)
        patient_id = filename.split('_slice')[0]
        patients[patient_id]['slices'].append(filename)

        # Check if this slice has tumor (class 2)
        label = np.load(label_path)
        if np.any(label == 2):
            patients[patient_id]['has_tumor'] = True

    return dict(patients)


def create_combined_splits():
    print("=" * 60)
    print("  CREATING COMBINED MSD + PANORAMA SPLITS")
    print("=" * 60)

    # Gather patients from both datasets
    print("\nScanning MSD dataset...")
    msd_patients = gather_patient_slices(MSD_IMAGES_DIR, MSD_LABELS_DIR, "MSD")
    print(f"  Found {len(msd_patients)} patients, "
          f"{sum(len(p['slices']) for p in msd_patients.values())} slices")

    print("Scanning PANORAMA dataset...")
    pan_patients = gather_patient_slices(PANORAMA_IMAGES_DIR, PANORAMA_LABELS_DIR, "PANORAMA")
    print(f"  Found {len(pan_patients)} patients, "
          f"{sum(len(p['slices']) for p in pan_patients.values())} slices")

    # Check for patient ID collisions between datasets
    msd_ids = set(msd_patients.keys())
    pan_ids = set(pan_patients.keys())
    collisions = msd_ids & pan_ids
    if collisions:
        # Prefix PANORAMA patient IDs to avoid collision
        print(f"\n  WARNING: {len(collisions)} patient ID collisions detected. "
              f"Prefixing PANORAMA IDs with 'pan_'")
        new_pan = {}
        for pid, info in pan_patients.items():
            new_pid = f"pan_{pid}" if pid in collisions else pid
            # Rename slice files conceptually (they stay on disk with original names)
            new_pan[new_pid] = info
        pan_patients = new_pan

    # Stratified split: separate tumor-positive and tumor-negative patients per dataset
    random.seed(RANDOM_SEED)

    def stratified_split(patients_dict, dataset_name):
        """Split patients into train/val with tumor stratification."""
        tumor_patients = [pid for pid, info in patients_dict.items() if info['has_tumor']]
        normal_patients = [pid for pid, info in patients_dict.items() if not info['has_tumor']]

        random.shuffle(tumor_patients)
        random.shuffle(normal_patients)

        n_val_tumor = max(1, int(len(tumor_patients) * VAL_RATIO))
        n_val_normal = max(0, int(len(normal_patients) * VAL_RATIO))

        val_pids = tumor_patients[:n_val_tumor] + normal_patients[:n_val_normal]
        train_pids = tumor_patients[n_val_tumor:] + normal_patients[n_val_normal:]

        print(f"\n  {dataset_name} split:")
        print(f"    Train: {len(train_pids)} patients "
              f"({sum(1 for p in train_pids if patients_dict[p]['has_tumor'])} tumor, "
              f"{sum(1 for p in train_pids if not patients_dict[p]['has_tumor'])} normal)")
        print(f"    Val:   {len(val_pids)} patients "
              f"({sum(1 for p in val_pids if patients_dict[p]['has_tumor'])} tumor, "
              f"{sum(1 for p in val_pids if not patients_dict[p]['has_tumor'])} normal)")

        return train_pids, val_pids

    msd_train_pids, msd_val_pids = stratified_split(msd_patients, "MSD")
    pan_train_pids, pan_val_pids = stratified_split(pan_patients, "PANORAMA")

    # Collect all slice filenames, tracking which dataset they belong to
    train_files = []
    val_files = []

    # We need to track the source directory for each file since MSD and PANORAMA
    # slices are in different directories. We'll store tuples (filename, dataset).
    train_entries = []
    val_entries = []

    for pid in msd_train_pids:
        for fname in msd_patients[pid]['slices']:
            train_files.append(fname)
            train_entries.append({'file': fname, 'dataset': 'MSD', 'patient': pid})

    for pid in msd_val_pids:
        for fname in msd_patients[pid]['slices']:
            val_files.append(fname)
            val_entries.append({'file': fname, 'dataset': 'MSD', 'patient': pid})

    for pid in pan_train_pids:
        real_pid = pid.replace('pan_', '') if pid.startswith('pan_') else pid
        info = pan_patients[pid]
        for fname in info['slices']:
            train_files.append(fname)
            train_entries.append({'file': fname, 'dataset': 'PANORAMA', 'patient': pid})

    for pid in pan_val_pids:
        real_pid = pid.replace('pan_', '') if pid.startswith('pan_') else pid
        info = pan_patients[pid]
        for fname in info['slices']:
            val_files.append(fname)
            val_entries.append({'file': fname, 'dataset': 'PANORAMA', 'patient': pid})

    # Shuffle training data
    random.shuffle(train_files)

    # Save the split
    split_data = {
        'train': train_files,
        'val': val_files,
    }

    with open(OUTPUT_SPLIT, 'w') as f:
        json.dump(split_data, f, indent=2)

    # Save metadata
    meta = {
        'seed': RANDOM_SEED,
        'val_ratio': VAL_RATIO,
        'total_patients': len(msd_patients) + len(pan_patients),
        'total_slices': len(train_files) + len(val_files),
        'train': {
            'total_slices': len(train_files),
            'msd_patients': len(msd_train_pids),
            'msd_slices': sum(len(msd_patients[p]['slices']) for p in msd_train_pids),
            'panorama_patients': len(pan_train_pids),
            'panorama_slices': sum(len(pan_patients[p]['slices']) for p in pan_train_pids),
        },
        'val': {
            'total_slices': len(val_files),
            'msd_patients': len(msd_val_pids),
            'msd_slices': sum(len(msd_patients[p]['slices']) for p in msd_val_pids),
            'panorama_patients': len(pan_val_pids),
            'panorama_slices': sum(len(pan_patients[p]['slices']) for p in pan_val_pids),
        },
        'msd_patients': {
            'train': sorted(msd_train_pids),
            'val': sorted(msd_val_pids),
        },
        'panorama_patients': {
            'train': sorted(pan_train_pids),
            'val': sorted(pan_val_pids),
        }
    }

    with open(OUTPUT_META, 'w') as f:
        json.dump(meta, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("  COMBINED SPLIT SUMMARY")
    print("=" * 60)
    print(f"\n  Train: {len(train_files)} slices "
          f"({meta['train']['msd_slices']} MSD + {meta['train']['panorama_slices']} PANORAMA)")
    print(f"  Val:   {len(val_files)} slices "
          f"({meta['val']['msd_slices']} MSD + {meta['val']['panorama_slices']} PANORAMA)")
    print(f"\n  Saved split to: {OUTPUT_SPLIT}")
    print(f"  Saved meta to:  {OUTPUT_META}")

    return split_data, meta


if __name__ == "__main__":
    create_combined_splits()
