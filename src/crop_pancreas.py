# src/crop_pancreas.py
import os
import json
import numpy as np
import nibabel as nib
from preprocessing import apply_window
from config import DATA_ROOT

OUTPUT_DIR = "../data/processed_cropped"
MARGIN = 20  # extra pixels around the pancreas, so we don't crop too tightly

def get_bounding_box(label_slice, margin=MARGIN):
    """Find the rectangular region containing the pancreas/cancer in a 2D label slice."""
    rows = np.any(label_slice > 0, axis=1)
    cols = np.any(label_slice > 0, axis=0)

    if not rows.any():  # no pancreas in this slice at all
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = label_slice.shape
    rmin = max(0, rmin - margin)
    rmax = min(h, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(w, cmax + margin)

    return rmin, rmax, cmin, cmax


def extract_cropped_slices():
    with open(f"{DATA_ROOT}/dataset.json") as f:
        dataset_info = json.load(f)

    training_pairs = dataset_info["training"]

    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

    total_saved = 0

    for i, pair in enumerate(training_pairs):
        image_path = f"{DATA_ROOT}/{pair['image'][2:]}"
        label_path = f"{DATA_ROOT}/{pair['label'][2:]}"
        patient_id = os.path.basename(image_path).replace('.nii.gz', '')

        image_data = nib.load(image_path).get_fdata()
        label_data = nib.load(label_path).get_fdata()

        useful_slices = np.where(np.any(label_data > 0, axis=(0, 1)))[0]

        for slice_idx in useful_slices:
            ct_slice = image_data[:, :, slice_idx]
            label_slice = label_data[:, :, slice_idx]

            bbox = get_bounding_box(label_slice)
            if bbox is None:
                continue
            rmin, rmax, cmin, cmax = bbox

            # Crop both image and label to just the pancreas region
            cropped_ct = ct_slice[rmin:rmax, cmin:cmax]
            cropped_label = label_slice[rmin:rmax, cmin:cmax]

            windowed = apply_window(cropped_ct).astype(np.float32)
            cropped_label = cropped_label.astype(np.uint8)

            slice_filename = f"{patient_id}_slice{slice_idx:03d}.npy"
            np.save(f"{OUTPUT_DIR}/images/{slice_filename}", windowed)
            np.save(f"{OUTPUT_DIR}/labels/{slice_filename}", cropped_label)
            total_saved += 1

        print(f"[{i+1}/{len(training_pairs)}] {patient_id}: cropped slices saved")

    print(f"\nDone. Total cropped slices saved: {total_saved}")

if __name__ == "__main__":
    extract_cropped_slices()