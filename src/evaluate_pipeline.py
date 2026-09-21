# src/evaluate_pipeline.py
"""
Quantitative evaluation of the full two-stage PancreaScan pipeline.

Runs the complete Stage 1 (localize) → Stage 2 (detect) pipeline on every
validation slice and computes:
  - Per-class Dice scores (background, pancreas, tumor)
  - Slice-level tumor detection sensitivity / specificity
  - Per-pixel confusion matrix (3×3)
  - Precision, recall, F1 per class
  - Per-patient aggregated Dice scores

Results are printed as formatted tables and saved to outputs/evaluation_results.json.
"""
import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm

from model import UNet
from pipeline import run_stage1, run_stage2, combine_predictions
from preprocessing import apply_window


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

STAGE1_MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "best_model_stage1.pth")
STAGE2_MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "best_model_stage2.pth")
DEFAULT_SPLIT_FILE = os.path.join(PROJECT_DIR, "data", "processed", "split.json")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_DIR, "outputs", "evaluation_results.json")

DEFAULT_IMAGE_DIRS = [
    os.path.join(PROJECT_DIR, "data", "processed", "images"),
    os.path.join(PROJECT_DIR, "data", "panorama_processed", "images"),
]
DEFAULT_LABEL_DIRS = [
    os.path.join(PROJECT_DIR, "data", "processed", "labels"),
    os.path.join(PROJECT_DIR, "data", "panorama_processed", "labels"),
]

def find_file(filename, dirs):
    for d in dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------
def dice_score(pred, target, class_id, epsilon=1e-6):
    """Dice coefficient for a single class."""
    pred_mask   = (pred == class_id).astype(np.float64)
    target_mask = (target == class_id).astype(np.float64)

    intersection = (pred_mask * target_mask).sum()
    total        = pred_mask.sum() + target_mask.sum()

    return float((2 * intersection + epsilon) / (total + epsilon))


def compute_confusion_matrix(all_preds, all_labels, num_classes=3):
    """
    Build a num_classes × num_classes confusion matrix.
    cm[i][j] = number of pixels with true class i predicted as class j.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_cls in range(num_classes):
        for pred_cls in range(num_classes):
            cm[true_cls, pred_cls] = int(
                np.sum((all_labels == true_cls) & (all_preds == pred_cls))
            )
    return cm


def precision_recall_f1_from_cm(cm):
    """Compute per-class precision, recall, F1 from confusion matrix."""
    num_classes = cm.shape[0]
    metrics = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp       # other classes predicted as c
        fn = cm[c, :].sum() - tp       # class c predicted as other classes

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[c] = {
            'precision': round(precision, 4),
            'recall':    round(recall, 4),
            'f1':        round(f1, 4),
            'support':   int(cm[c, :].sum()),
        }
    return metrics


# -------------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------------
def evaluate(split_file=None, output_path=None, limit=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if split_file is None:
        split_file = DEFAULT_SPLIT_FILE
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH

    print(f"[Eval] Using split file: {split_file}")
    print(f"[Eval] Saving output to: {output_path}")

    # Check models exist
    for path, name in [(STAGE1_MODEL_PATH, "Stage 1"), (STAGE2_MODEL_PATH, "Stage 2")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} model not found at {path}")
            sys.exit(1)

    # Load models
    model_s1 = UNet(in_channels=1, num_classes=2).to(device)
    model_s1.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=device, weights_only=True))
    model_s1.eval()
    print("[Eval] Stage 1 model loaded.")

    model_s2 = UNet(in_channels=1, num_classes=3).to(device)
    model_s2.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location=device, weights_only=True))
    model_s2.eval()
    print("[Eval] Stage 2 model loaded.")

    # Load validation filenames
    with open(split_file) as f:
        val_files = json.load(f)['val']

    if limit is not None and limit > 0:
        val_files = val_files[:limit]
        print(f"[Eval] Limiting evaluation to {limit} slices.")

    print(f"[Eval] Evaluating {len(val_files)} validation slices...\n")

    # Storage for metrics
    class_names = {0: 'background', 1: 'pancreas', 2: 'tumor'}
    dice_per_slice = {0: [], 1: [], 2: []}

    # Per-patient tracking
    patient_dice = {}  # patient_id -> {class_id: [dice_scores]}

    # Slice-level tumor detection (binary classification)
    slice_tumor_tp = 0   # ground truth has tumor AND model detected tumor
    slice_tumor_tn = 0   # no tumor in GT AND model didn't detect tumor
    slice_tumor_fp = 0   # no tumor in GT BUT model detected tumor
    slice_tumor_fn = 0   # tumor in GT BUT model missed it

    # Collect all pixel predictions for global confusion matrix
    all_preds_flat  = []
    all_labels_flat = []

    for filename in tqdm(val_files, desc="Two-stage evaluation"):
        image_path = find_file(filename, DEFAULT_IMAGE_DIRS)
        label_path = find_file(filename, DEFAULT_LABEL_DIRS)

        if not image_path or not label_path:
            continue

        image_np = np.load(image_path)
        label_np = np.load(label_path).astype(int)

        # Run full two-stage pipeline
        stage1_pred, bbox = run_stage1(image_np, model_s1, device)

        if bbox is None:
            # No pancreas detected — prediction is all-background
            full_pred = np.zeros(image_np.shape, dtype=int)
        else:
            stage2_pred_crop, stage2_probs_crop, crop = run_stage2(image_np, bbox, model_s2, device)
            full_pred = combine_predictions(image_np, stage1_pred, stage2_pred_crop, bbox)

        # Resize label to match prediction shape (they should already match, but safety)
        if full_pred.shape != label_np.shape:
            from torch.nn.functional import interpolate
            lbl_t = torch.from_numpy(label_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            lbl_resized = interpolate(lbl_t, size=full_pred.shape, mode='nearest')
            label_np = lbl_resized.squeeze().numpy().astype(int)

        # Per-class Dice
        for cls_id in [0, 1, 2]:
            d = dice_score(full_pred, label_np, cls_id)
            dice_per_slice[cls_id].append(d)

        # Per-patient Dice tracking
        patient_id = filename.split('_slice')[0]
        if patient_id not in patient_dice:
            patient_dice[patient_id] = {0: [], 1: [], 2: []}
        for cls_id in [0, 1, 2]:
            patient_dice[patient_id][cls_id].append(
                dice_score(full_pred, label_np, cls_id)
            )

        # Slice-level tumor detection
        gt_has_tumor   = bool(np.any(label_np == 2))
        pred_has_tumor = bool(np.any(full_pred == 2))

        if gt_has_tumor and pred_has_tumor:
            slice_tumor_tp += 1
        elif not gt_has_tumor and not pred_has_tumor:
            slice_tumor_tn += 1
        elif not gt_has_tumor and pred_has_tumor:
            slice_tumor_fp += 1
        else:
            slice_tumor_fn += 1

        # Flatten for global confusion matrix (subsample to keep memory sane)
        # Take every 4th pixel to reduce memory for very large datasets
        step = 4
        all_preds_flat.append(full_pred.flatten()[::step])
        all_labels_flat.append(label_np.flatten()[::step])

    # ---- Aggregate results ----
    print("\n" + "=" * 60)
    print("  PANCREASCAN — TWO-STAGE PIPELINE EVALUATION RESULTS")
    print("=" * 60)

    # 1. Per-class Dice scores
    print("\n--- Per-Class Dice Scores (mean ± std across slices) ---")
    dice_results = {}
    for cls_id in [0, 1, 2]:
        scores = dice_per_slice[cls_id]
        mean_d = np.mean(scores)
        std_d  = np.std(scores)
        print(f"  {class_names[cls_id]:12s}: {mean_d:.4f} ± {std_d:.4f}  (n={len(scores)})")
        dice_results[class_names[cls_id]] = {
            'mean': round(float(mean_d), 4),
            'std':  round(float(std_d), 4),
            'n':    len(scores),
        }

    # 2. Per-patient Dice (averaged across each patient's slices, then across patients)
    print("\n--- Per-Patient Dice Scores (patient-level average) ---")
    patient_avg_dice = {0: [], 1: [], 2: []}
    for pid, cls_dict in patient_dice.items():
        for cls_id in [0, 1, 2]:
            if cls_dict[cls_id]:
                patient_avg_dice[cls_id].append(np.mean(cls_dict[cls_id]))

    patient_dice_results = {}
    for cls_id in [0, 1, 2]:
        scores = patient_avg_dice[cls_id]
        mean_d = np.mean(scores)
        std_d  = np.std(scores)
        print(f"  {class_names[cls_id]:12s}: {mean_d:.4f} ± {std_d:.4f}  (n_patients={len(scores)})")
        patient_dice_results[class_names[cls_id]] = {
            'mean': round(float(mean_d), 4),
            'std':  round(float(std_d), 4),
            'n_patients': len(scores),
        }

    # 3. Slice-level tumor detection (sensitivity / specificity)
    total_slices = slice_tumor_tp + slice_tumor_tn + slice_tumor_fp + slice_tumor_fn
    sensitivity = slice_tumor_tp / (slice_tumor_tp + slice_tumor_fn) if (slice_tumor_tp + slice_tumor_fn) > 0 else 0.0
    specificity = slice_tumor_tn / (slice_tumor_tn + slice_tumor_fp) if (slice_tumor_tn + slice_tumor_fp) > 0 else 0.0
    accuracy    = (slice_tumor_tp + slice_tumor_tn) / total_slices if total_slices > 0 else 0.0

    print("\n--- Slice-Level Tumor Detection ---")
    print(f"  True Positives:  {slice_tumor_tp}")
    print(f"  True Negatives:  {slice_tumor_tn}")
    print(f"  False Positives: {slice_tumor_fp}")
    print(f"  False Negatives: {slice_tumor_fn}")
    print(f"  Sensitivity:     {sensitivity:.4f}")
    print(f"  Specificity:     {specificity:.4f}")
    print(f"  Accuracy:        {accuracy:.4f}")

    detection_results = {
        'true_positives':  slice_tumor_tp,
        'true_negatives':  slice_tumor_tn,
        'false_positives': slice_tumor_fp,
        'false_negatives': slice_tumor_fn,
        'sensitivity':     round(sensitivity, 4),
        'specificity':     round(specificity, 4),
        'accuracy':        round(accuracy, 4),
    }

    # 4. Global confusion matrix
    all_preds_arr  = np.concatenate(all_preds_flat)
    all_labels_arr = np.concatenate(all_labels_flat)
    cm = compute_confusion_matrix(all_preds_arr, all_labels_arr, num_classes=3)

    print("\n--- Pixel-Level Confusion Matrix ---")
    print(f"  {'':12s}  {'Pred BG':>12s}  {'Pred Panc':>12s}  {'Pred Tumor':>12s}")
    for i in range(3):
        print(f"  {class_names[i]:12s}  {cm[i,0]:>12,d}  {cm[i,1]:>12,d}  {cm[i,2]:>12,d}")

    # 5. Per-class precision / recall / F1
    prf = precision_recall_f1_from_cm(cm)
    print("\n--- Per-Class Precision / Recall / F1 ---")
    print(f"  {'Class':12s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'Support':>10s}")
    prf_results = {}
    for cls_id in [0, 1, 2]:
        m = prf[cls_id]
        print(f"  {class_names[cls_id]:12s}  {m['precision']:>10.4f}  {m['recall']:>10.4f}  {m['f1']:>10.4f}  {m['support']:>10,d}")
        prf_results[class_names[cls_id]] = m

    print("\n" + "=" * 60)

    # ---- Save to JSON ----
    results = {
        'pipeline': 'Two-Stage (Stage 1: localize, Stage 2: detect)',
        'dataset': 'Medical Segmentation Decathlon — Task07_Pancreas',
        'split': 'validation',
        'num_slices': len(val_files),
        'num_patients': len(patient_dice),
        'dice_per_slice': dice_results,
        'dice_per_patient': patient_dice_results,
        'slice_tumor_detection': detection_results,
        'confusion_matrix': {
            'classes': ['background', 'pancreas', 'tumor'],
            'matrix': cm.tolist(),
            'note': 'Subsampled (every 4th pixel) to manage memory',
        },
        'classification_report': prf_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate two-stage PancreaScan pipeline")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Path to split.json or split_combined.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output JSON")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of slices")
    args = parser.parse_args()
    evaluate(split_file=args.split_file, output_path=args.output, limit=args.limit)
