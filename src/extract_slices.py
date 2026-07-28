# src/extract_slices.py
import os
import json
import numpy as np
import nibabel as nib
from preprocessing import apply_window
from config import DATA_ROOT

OUTPUT_DIR = "../data/processed"

def extract_all_slices():
    # Load the verified image/label pairings from dataset.json
    with open(f"{DATA_ROOT}/dataset.json") as f:
        dataset_info = json.load(f)

    training_pairs = dataset_info["training"]

    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

    total_slices_saved = 0

    for i, pair in enumerate(training_pairs):
        image_path = f"{DATA_ROOT}/{pair['image'][2:]}"  # strip leading './'
        label_path = f"{DATA_ROOT}/{pair['label'][2:]}"

        patient_id = os.path.basename(image_path).replace('.nii.gz', '')

        img = nib.load(image_path)
        lbl = nib.load(label_path)

        image_data = img.get_fdata()
        label_data = lbl.get_fdata()

        # Find slices where pancreas (1) or cancer (2) is present
        useful_slices = np.where(np.any(label_data > 0, axis=(0, 1)))[0]

        for slice_idx in useful_slices:
            ct_slice = image_data[:, :, slice_idx]
            label_slice = label_data[:, :, slice_idx]

            windowed = apply_window(ct_slice).astype(np.float32)
            label_slice = label_slice.astype(np.uint8)

            slice_filename = f"{patient_id}_slice{slice_idx:03d}.npy"
            np.save(f"{OUTPUT_DIR}/images/{slice_filename}", windowed)
            np.save(f"{OUTPUT_DIR}/labels/{slice_filename}", label_slice)

            total_slices_saved += 1

        print(f"[{i+1}/{len(training_pairs)}] {patient_id}: "
              f"{len(useful_slices)} slices saved")

    print(f"\nDone. Total slices saved: {total_slices_saved}")

if __name__ == "__main__":
    extract_all_slices()