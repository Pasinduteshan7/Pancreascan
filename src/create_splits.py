# src/create_splits.py
import os
import json
import random

IMAGES_DIR = "../data/processed/images"

def create_patient_split(val_ratio=0.15, seed=42):
    all_files = os.listdir(IMAGES_DIR)

    # Extract unique patient IDs from filenames like 'pancreas_001_slice044.npy'
    patient_ids = sorted(set(f.split('_slice')[0] for f in all_files))
    print(f"Total unique patients: {len(patient_ids)}")

    random.seed(seed)  # reproducible split
    random.shuffle(patient_ids)

    n_val = int(len(patient_ids) * val_ratio)
    val_patients = set(patient_ids[:n_val])
    train_patients = set(patient_ids[n_val:])

    train_files = [f for f in all_files if f.split('_slice')[0] in train_patients]
    val_files = [f for f in all_files if f.split('_slice')[0] in val_patients]

    print(f"Train patients: {len(train_patients)} -> {len(train_files)} slices")
    print(f"Val patients: {len(val_patients)} -> {len(val_files)} slices")

    split = {"train": train_files, "val": val_files}
    with open("../data/processed/split.json", "w") as f:
        json.dump(split, f, indent=2)

    print("Saved split.json")

if __name__ == "__main__":
    create_patient_split()