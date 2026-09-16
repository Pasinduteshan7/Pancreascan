# src/crop_panorama.py
"""
Generate pancreas-cropped slices from the extracted PANORAMA data for Stage 2 training.

Uses the same bounding-box cropping approach as crop_pancreas.py but operates on
the already-extracted 2D .npy slices from panorama_processed/ instead of raw NIfTI volumes.
"""
import os
import sys
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INPUT_IMAGES  = os.path.join(PROJECT_DIR, "data", "panorama_processed", "images")
INPUT_LABELS  = os.path.join(PROJECT_DIR, "data", "panorama_processed", "labels")
OUTPUT_IMAGES = os.path.join(PROJECT_DIR, "data", "panorama_cropped", "images")
OUTPUT_LABELS = os.path.join(PROJECT_DIR, "data", "panorama_cropped", "labels")

MARGIN = 20


def get_bounding_box(label_slice, margin=MARGIN):
    """Find the rectangular region containing the pancreas/cancer in a 2D label slice."""
    rows = np.any(label_slice > 0, axis=1)
    cols = np.any(label_slice > 0, axis=0)

    if not rows.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = label_slice.shape
    rmin = max(0, rmin - margin)
    rmax = min(h, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(w, cmax + margin)

    return rmin, rmax, cmin, cmax


def crop_panorama_slices():
    if not os.path.exists(INPUT_IMAGES):
        print(f"ERROR: Input directory not found: {INPUT_IMAGES}")
        print("Run extract_panorama.py first.")
        sys.exit(1)

    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    os.makedirs(OUTPUT_LABELS, exist_ok=True)

    files = sorted([f for f in os.listdir(INPUT_IMAGES) if f.endswith('.npy')])
    print(f"Cropping {len(files)} PANORAMA slices...")

    total_saved = 0
    skipped = 0

    for i, filename in enumerate(files):
        image = np.load(os.path.join(INPUT_IMAGES, filename))
        label = np.load(os.path.join(INPUT_LABELS, filename))

        bbox = get_bounding_box(label)
        if bbox is None:
            skipped += 1
            continue

        rmin, rmax, cmin, cmax = bbox
        cropped_image = image[rmin:rmax, cmin:cmax].astype(np.float32)
        cropped_label = label[rmin:rmax, cmin:cmax].astype(np.uint8)

        np.save(os.path.join(OUTPUT_IMAGES, filename), cropped_image)
        np.save(os.path.join(OUTPUT_LABELS, filename), cropped_label)
        total_saved += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(files)}] processed...")

    print(f"\nDone. Cropped slices saved: {total_saved}, skipped: {skipped}")
    print(f"Output: {OUTPUT_IMAGES}")


if __name__ == "__main__":
    crop_panorama_slices()
