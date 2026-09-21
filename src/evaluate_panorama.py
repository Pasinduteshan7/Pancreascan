# src/evaluate_panorama.py
"""
Generalization evaluation of the MSD-trained PancreaScan pipeline on PANORAMA dataset.

Tests the two-stage model trained on Medical Segmentation Decathlon (MSD)
directly on external CT scans from the PANORAMA challenge WITHOUT retraining.
This provides the definitive "out-of-domain generalization" metric for the research.

Computes:
  - Per-class Dice scores (background, pancreas, tumor)
  - Slice-level tumor detection sensitivity / specificity / accuracy
  - Patient-level tumor detection
  - Direct side-by-side comparison with MSD in-domain validation performance

Results are printed and saved to outputs/evaluation_panorama.json.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

from model import UNet
from pipeline import run_stage1, run_stage2, combine_predictions

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

STAGE1_MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "best_model_stage1.pth")
STAGE2_MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "best_model_stage2.pth")
PANORAMA_DIR      = os.path.join(PROJECT_DIR, "data", "panorama_processed")
MSD_RESULTS_PATH  = os.path.join(PROJECT_DIR, "outputs", "evaluation_results.json")
OUTPUT_PATH       = os.path.join(PROJECT_DIR, "outputs", "evaluation_panorama.json")


def dice_score(pred, target, class_id, epsilon=1e-6):
    """Dice coefficient for a single class."""
    pred_mask   = (pred == class_id).astype(np.float64)
    target_mask = (target == class_id).astype(np.float64)

    intersection = (pred_mask * target_mask).sum()
    total        = pred_mask.sum() + target_mask.sum()

    return float((2 * intersection + epsilon) / (total + epsilon))


def evaluate_panorama(limit=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check models
    for path, name in [(STAGE1_MODEL_PATH, "Stage 1"), (STAGE2_MODEL_PATH, "Stage 2")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} model not found at {path}")
            sys.exit(1)

    images_dir = os.path.join(PANORAMA_DIR, "images")
    labels_dir = os.path.join(PANORAMA_DIR, "labels")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"ERROR: PANORAMA processed directory not found at {PANORAMA_DIR}")
        print("Please run `python extract_panorama.py` first to extract slices.")
        sys.exit(1)

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
    if len(image_files) == 0:
        print(f"ERROR: No .npy slices found in {images_dir}")
        sys.exit(1)

    if limit is not None and limit > 0:
        image_files = image_files[:limit]
        print(f"[PANORAMA Eval] Running on subset of {len(image_files)} slices (--limit {limit}).")
    else:
        print(f"[PANORAMA Eval] Evaluating all {len(image_files)} PANORAMA slices...")

    # Load models
    model_s1 = UNet(in_channels=1, num_classes=2).to(device)
    model_s1.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=device, weights_only=True))
    model_s1.eval()

    model_s2 = UNet(in_channels=1, num_classes=3).to(device)
    model_s2.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location=device, weights_only=True))
    model_s2.eval()

    class_names = {0: 'background', 1: 'pancreas', 2: 'tumor'}
    dice_per_slice = {0: [], 1: [], 2: []}
    patient_dice = {}

    slice_tumor_tp = 0
    slice_tumor_tn = 0
    slice_tumor_fp = 0
    slice_tumor_fn = 0

    patient_tumor_gt = {}
    patient_tumor_pred = {}

    for filename in tqdm(image_files, desc="PANORAMA Generalization Eval"):
        img_path = os.path.join(images_dir, filename)
        lbl_path = os.path.join(labels_dir, filename)

        if not os.path.exists(lbl_path):
            continue

        image_np = np.load(img_path)
        label_np = np.load(lbl_path).astype(int)

        patient_id = filename.split('_slice')[0]
        if patient_id not in patient_dice:
            patient_dice[patient_id] = {0: [], 1: [], 2: []}
            patient_tumor_gt[patient_id] = False
            patient_tumor_pred[patient_id] = False

        # Run two-stage inference
        stage1_pred, bbox = run_stage1(image_np, model_s1, device)

        if bbox is None:
            full_pred = np.zeros(image_np.shape, dtype=int)
        else:
            stage2_pred_crop, _, _ = run_stage2(image_np, bbox, model_s2, device)
            full_pred = combine_predictions(image_np, stage1_pred, stage2_pred_crop, bbox)

        if full_pred.shape != label_np.shape:
            from torch.nn.functional import interpolate
            lbl_t = torch.from_numpy(label_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            lbl_resized = interpolate(lbl_t, size=full_pred.shape, mode='nearest')
            label_np = lbl_resized.squeeze().numpy().astype(int)

        # Per-class Dice
        for cls_id in [0, 1, 2]:
            d = dice_score(full_pred, label_np, cls_id)
            dice_per_slice[cls_id].append(d)
            patient_dice[patient_id][cls_id].append(d)

        # Slice-level tumor detection
        gt_has_tumor = bool(np.any(label_np == 2))
        pred_has_tumor = bool(np.any(full_pred == 2))

        if gt_has_tumor:
            patient_tumor_gt[patient_id] = True
        if pred_has_tumor:
            patient_tumor_pred[patient_id] = True

        if gt_has_tumor and pred_has_tumor:
            slice_tumor_tp += 1
        elif not gt_has_tumor and not pred_has_tumor:
            slice_tumor_tn += 1
        elif not gt_has_tumor and pred_has_tumor:
            slice_tumor_fp += 1
        else:
            slice_tumor_fn += 1

    # Aggregates
    total_slices = slice_tumor_tp + slice_tumor_tn + slice_tumor_fp + slice_tumor_fn
    sensitivity = slice_tumor_tp / (slice_tumor_tp + slice_tumor_fn) if (slice_tumor_tp + slice_tumor_fn) > 0 else 0.0
    specificity = slice_tumor_tn / (slice_tumor_tn + slice_tumor_fp) if (slice_tumor_tn + slice_tumor_fp) > 0 else 0.0
    accuracy = (slice_tumor_tp + slice_tumor_tn) / total_slices if total_slices > 0 else 0.0
    precision = slice_tumor_tp / (slice_tumor_tp + slice_tumor_fp) if (slice_tumor_tp + slice_tumor_fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    # Patient-level metrics
    p_tp = sum(1 for pid in patient_tumor_gt if patient_tumor_gt[pid] and patient_tumor_pred[pid])
    p_tn = sum(1 for pid in patient_tumor_gt if not patient_tumor_gt[pid] and not patient_tumor_pred[pid])
    p_fp = sum(1 for pid in patient_tumor_gt if not patient_tumor_gt[pid] and patient_tumor_pred[pid])
    p_fn = sum(1 for pid in patient_tumor_gt if patient_tumor_gt[pid] and not patient_tumor_pred[pid])
    p_sens = p_tp / (p_tp + p_fn) if (p_tp + p_fn) > 0 else 0.0
    p_spec = p_tn / (p_tn + p_fp) if (p_tn + p_fp) > 0 else 0.0

    print("\n" + "=" * 65)
    print("  PANCREASCAN — PANORAMA GENERALIZATION EVALUATION RESULTS")
    print("=" * 65)

    print("\n--- Per-Class Dice Scores (mean ± std across slices) ---")
    dice_results = {}
    for cls_id in [0, 1, 2]:
        scores = dice_per_slice[cls_id]
        mean_d = np.mean(scores) if scores else 0.0
        std_d  = np.std(scores) if scores else 0.0
        print(f"  {class_names[cls_id]:12s}: {mean_d:.4f} ± {std_d:.4f}  (n={len(scores)})")
        dice_results[class_names[cls_id]] = {'mean': round(float(mean_d), 4), 'std': round(float(std_d), 4)}

    print("\n--- Slice-Level Tumor Detection ---")
    print(f"  Evaluated Slices: {total_slices}")
    print(f"  True Positives:   {slice_tumor_tp}")
    print(f"  True Negatives:   {slice_tumor_tn}")
    print(f"  False Positives:  {slice_tumor_fp}")
    print(f"  False Negatives:  {slice_tumor_fn}")
    print(f"  Sensitivity:      {sensitivity * 100:.2f}%")
    print(f"  Specificity:      {specificity * 100:.2f}%")
    print(f"  Accuracy:         {accuracy * 100:.2f}%")
    print(f"  Precision:        {precision * 100:.2f}%")
    print(f"  F1 Score:         {f1:.4f}")

    print("\n--- Patient-Level Detection ---")
    print(f"  Total Patients:   {len(patient_tumor_gt)}")
    print(f"  Sensitivity:      {p_sens * 100:.2f}% ({p_tp}/{p_tp + p_fn})")
    print(f"  Specificity:      {p_spec * 100:.2f}% ({p_tn}/{p_tn + p_fp})")

    # Compare with MSD baseline if available
    msd_comparison = None
    if os.path.exists(MSD_RESULTS_PATH):
        try:
            with open(MSD_RESULTS_PATH) as f:
                msd_res = json.load(f)
            print("\n" + "=" * 65)
            print("  CROSS-DATASET GENERALIZATION COMPARISON")
            print("=" * 65)
            print(f"{'Metric':<25} | {'MSD (In-Domain)':<18} | {'PANORAMA (Generalization)':<22}")
            print("-" * 72)

            msd_pancreas = msd_res['dice_per_slice']['pancreas']['mean']
            pan_pancreas = dice_results['pancreas']['mean']
            print(f"{'Pancreas Dice':<25} | {msd_pancreas:<18.4f} | {pan_pancreas:<22.4f}")

            msd_tumor = msd_res['dice_per_slice']['tumor']['mean']
            pan_tumor = dice_results['tumor']['mean']
            print(f"{'Tumor Dice':<25} | {msd_tumor:<18.4f} | {pan_tumor:<22.4f}")

            msd_sens = msd_res['slice_tumor_detection']['sensitivity']
            print(f"{'Tumor Sensitivity':<25} | {msd_sens * 100:<17.1f}% | {sensitivity * 100:<21.1f}%")

            msd_spec = msd_res['slice_tumor_detection']['specificity']
            print(f"{'Tumor Specificity':<25} | {msd_spec * 100:<17.1f}% | {specificity * 100:<21.1f}%")

            msd_comparison = {
                'in_domain_msd': msd_res,
                'delta_pancreas_dice': round(pan_pancreas - msd_pancreas, 4),
                'delta_tumor_dice': round(pan_tumor - msd_tumor, 4),
            }
        except Exception as e:
            print(f"Could not load MSD baseline for comparison: {e}")

    # Save to JSON
    results = {
        'total_slices': total_slices,
        'per_class_dice': dice_results,
        'slice_tumor_detection': {
            'tp': slice_tumor_tp,
            'tn': slice_tumor_tn,
            'fp': slice_tumor_fp,
            'fn': slice_tumor_fn,
            'sensitivity': round(sensitivity, 4),
            'specificity': round(specificity, 4),
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'f1': round(f1, 4),
        },
        'patient_tumor_detection': {
            'total_patients': len(patient_tumor_gt),
            'tp': p_tp,
            'tn': p_tn,
            'fp': p_fp,
            'fn': p_fn,
            'sensitivity': round(p_sens, 4),
            'specificity': round(p_spec, 4),
        },
        'msd_comparison': msd_comparison
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Done] Full evaluation saved to: {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on PANORAMA dataset")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of slices to evaluate (e.g. for quick test)")
    args = parser.parse_args()
    evaluate_panorama(limit=args.limit)
