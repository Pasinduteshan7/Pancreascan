# src/extract_panorama.py
"""
Extract 2D slices from PANORAMA challenge CT volumes for generalization testing.

Maps PANORAMA labels to the MSD convention used by PancreaScan:
  PANORAMA 0 (Background)           → 0 (background)
  PANORAMA 1 (PDAC lesion)          → 2 (tumor)
  PANORAMA 2 (Veins)                → 0 (background)
  PANORAMA 3 (Arteries)             → 0 (background)
  PANORAMA 4 (Pancreas parenchyma)  → 1 (pancreas)
  PANORAMA 5 (Pancreatic duct)      → 1 (pancreas)
  PANORAMA 6 (Common bile duct)     → 0 (background)

Usage:
    python extract_panorama.py
    python extract_panorama.py --manual-only   # Only process manually-labeled cases (faster)
"""
import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
from preprocessing import apply_window

# -------------------------------------------------------------------------
# Paths — update these to match your local setup
# -------------------------------------------------------------------------
SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR         = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PANORAMA_IMAGES_DIR = r"F:\PANORAMA CHALANGE DATASET\batch_4\batch_4"
MANUAL_LABELS_DIR   = r"d:\PROJECTS\PERSONAL\myself\Pancreascan\panorama_labels\manual_labels"
AUTO_LABELS_DIR     = r"d:\PROJECTS\PERSONAL\myself\Pancreascan\panorama_labels\automatic_labels"
OUTPUT_DIR          = os.path.join(PROJECT_DIR, "data", "panorama_processed")


# -------------------------------------------------------------------------
# Label remapping
# -------------------------------------------------------------------------
def remap_panorama_label(label_slice):
    """
    Remap PANORAMA label values to MSD convention.

    PANORAMA: 0=bg, 1=PDAC, 2=veins, 3=arteries, 4=pancreas, 5=duct, 6=bile duct
    MSD:      0=bg, 1=pancreas, 2=tumor
    """
    remapped = np.zeros_like(label_slice, dtype=np.uint8)

    # Pancreas parenchyma (4) and pancreatic duct (5) → class 1 (pancreas)
    remapped[(label_slice == 4) | (label_slice == 5)] = 1

    # PDAC lesion (1) → class 2 (tumor)
    remapped[label_slice == 1] = 2

    # Everything else (0=bg, 2=veins, 3=arteries, 6=bile duct) stays 0

    return remapped


# -------------------------------------------------------------------------
# Main extraction
# -------------------------------------------------------------------------
def extract_panorama(manual_only=False, limit=None):
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

    # Build the list of image files and their matching labels
    image_files = sorted(os.listdir(PANORAMA_IMAGES_DIR))
    manual_labels = set(os.listdir(MANUAL_LABELS_DIR))
    auto_labels   = set(os.listdir(AUTO_LABELS_DIR))

    pairs = []
    for img_file in image_files:
        patient_id = img_file.replace('_0000.nii.gz', '')  # e.g., "101694_00001"
        label_name = f"{patient_id}.nii.gz"

        # Prefer manual labels (expert-annotated), fall back to automatic
        if label_name in manual_labels:
            label_path = os.path.join(MANUAL_LABELS_DIR, label_name)
            label_source = "manual"
        elif label_name in auto_labels:
            if manual_only:
                continue  # Skip non-manual when --manual-only
            label_path = os.path.join(AUTO_LABELS_DIR, label_name)
            label_source = "auto"
        else:
            print(f"  SKIP {patient_id}: no label found")
            continue

        pairs.append({
            'image_file': img_file,
            'label_path': label_path,
            'patient_id': patient_id,
            'label_source': label_source,
        })

    if limit is not None and limit > 0:
        pairs = pairs[:limit]
        print(f"Limiting to first {len(pairs)} cases (--limit {limit})")

    print(f"Found {len(pairs)} image-label pairs")
    if manual_only:
        print(f"  (manual-only mode: only expert-annotated PDAC cases)")
    print(f"  Manual labels: {sum(1 for p in pairs if p['label_source'] == 'manual')}")
    print(f"  Auto labels:   {sum(1 for p in pairs if p['label_source'] == 'auto')}")
    print()

    total_slices = 0
    metadata = []  # Track which patients have tumors for the generalization test

    for i, pair in enumerate(pairs):
        image_path = os.path.join(PANORAMA_IMAGES_DIR, pair['image_file'])
        patient_id = pair['patient_id']

        try:
            img_data = nib.load(image_path).get_fdata()
            lbl_data = nib.load(pair['label_path']).get_fdata()
        except Exception as e:
            print(f"  ERROR loading {patient_id}: {e}")
            continue

        # Check shape compatibility
        if img_data.shape != lbl_data.shape:
            print(f"  SKIP {patient_id}: shape mismatch img={img_data.shape} lbl={lbl_data.shape}")
            continue

        # Find slices where pancreas or tumor is present (using PANORAMA values 1, 4, 5)
        pancreas_mask = (lbl_data == 1) | (lbl_data == 4) | (lbl_data == 5)
        useful_slices = np.where(np.any(pancreas_mask, axis=(0, 1)))[0]

        if len(useful_slices) == 0:
            # For non-PDAC cases with auto labels, pancreas (4) should still be present
            # If nothing found, skip
            print(f"  SKIP {patient_id}: no pancreas/tumor in any slice")
            continue

        has_tumor = bool(np.any(lbl_data == 1))
        patient_slices = 0

        for slice_idx in useful_slices:
            ct_slice    = img_data[:, :, slice_idx]
            label_slice = lbl_data[:, :, slice_idx]

            # Apply windowing (same as MSD pipeline)
            windowed = apply_window(ct_slice).astype(np.float32)

            # Remap PANORAMA labels to MSD convention
            remapped = remap_panorama_label(label_slice)

            slice_filename = f"{patient_id}_slice{slice_idx:03d}.npy"
            np.save(f"{OUTPUT_DIR}/images/{slice_filename}", windowed)
            np.save(f"{OUTPUT_DIR}/labels/{slice_filename}", remapped)

            patient_slices += 1
            total_slices += 1

        metadata.append({
            'patient_id':   patient_id,
            'label_source': pair['label_source'],
            'has_tumor':    has_tumor,
            'num_slices':   patient_slices,
            'volume_shape': list(img_data.shape),
        })

        status = "🔴 PDAC" if has_tumor else "✅ Normal"
        print(f"[{i+1}/{len(pairs)}] {patient_id} ({pair['label_source']}): "
              f"{patient_slices} slices — {status}")

    # Save metadata
    meta = {
        'total_patients': len(metadata),
        'total_slices':   total_slices,
        'patients_with_tumor': sum(1 for m in metadata if m['has_tumor']),
        'patients_without_tumor': sum(1 for m in metadata if not m['has_tumor']),
        'patients': metadata,
    }
    with open(f"{OUTPUT_DIR}/panorama_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Total slices saved: {total_slices}")
    print(f"  Patients with PDAC:    {meta['patients_with_tumor']}")
    print(f"  Patients without PDAC: {meta['patients_without_tumor']}")
    print(f"  Metadata saved to: {OUTPUT_DIR}/panorama_metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PANORAMA slices")
    parser.add_argument("--manual-only", action="store_true",
                        help="Only process manually-labeled PDAC cases (faster)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of CT volumes to process (for quick testing)")
    args = parser.parse_args()
    extract_panorama(manual_only=args.manual_only, limit=args.limit)
