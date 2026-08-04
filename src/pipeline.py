# src/pipeline.py
"""
Two-Stage Inference Pipeline for PancreaScan

Stage 1 (Localization): Full-slice UNet finds the pancreas bounding box.
Stage 2 (Detection):    Zoomed-in UNet detects cancer within the cropped region.
Final output:           Stage 2 prediction mapped back onto the original full-slice coordinates.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import UNet
from preprocessing import apply_window

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
STAGE1_MODEL_PATH = "../outputs/best_model_stage1.pth"
STAGE2_MODEL_PATH = "../outputs/best_model_stage2.pth"
STAGE1_INPUT_SIZE = 256   # Stage 1 works on full slice resized to 256x256
STAGE2_INPUT_SIZE = 128   # Stage 2 works on the zoomed crop resized to 128x128
CROP_MARGIN = 20          # Extra pixels added around the Stage 1 bounding box


# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------
def load_models(device):
    """Load both stage models onto the given device."""
    model_s1 = UNet(in_channels=1, num_classes=2).to(device)
    model_s1.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=device, weights_only=True))
    model_s1.eval()
    print("[Pipeline] Stage 1 model loaded.")

    model_s2 = UNet(in_channels=1, num_classes=3).to(device)
    model_s2.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location=device, weights_only=True))
    model_s2.eval()
    print("[Pipeline] Stage 2 model loaded.")

    return model_s1, model_s2


# -------------------------------------------------------------------------
# Stage 1: Pancreas localization → bounding box
# -------------------------------------------------------------------------
def run_stage1(image_np, model_s1, device):
    """
    Run Stage 1 on a full CT slice (numpy array, already windowed 0-1).
    Returns:
        stage1_pred: (256, 256) binary mask (0=background, 1=pancreas-region)
        bbox: (rmin, rmax, cmin, cmax) in ORIGINAL image coordinates, or None if no pancreas found
    """
    original_h, original_w = image_np.shape

    # Resize to 256x256 for Stage 1
    tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    tensor_256 = F.interpolate(tensor, size=(STAGE1_INPUT_SIZE, STAGE1_INPUT_SIZE),
                                mode='bilinear', align_corners=False).to(device)

    with torch.no_grad():
        output = model_s1(tensor_256)              # (1, 2, 256, 256)
        pred = torch.argmax(output, dim=1)          # (1, 256, 256)
    stage1_pred = pred[0].cpu().numpy()             # (256, 256), values 0 or 1

    # Find bounding box in 256x256 space
    rows = np.any(stage1_pred > 0, axis=1)
    cols = np.any(stage1_pred > 0, axis=0)

    if not rows.any():
        return stage1_pred, None  # No pancreas detected → skip Stage 2

    rmin_256, rmax_256 = np.where(rows)[0][[0, -1]]
    cmin_256, cmax_256 = np.where(cols)[0][[0, -1]]

    # Scale bounding box back to original image coordinates
    scale_r = original_h / STAGE1_INPUT_SIZE
    scale_c = original_w / STAGE1_INPUT_SIZE

    rmin = int(max(0, (rmin_256 - CROP_MARGIN) * scale_r))
    rmax = int(min(original_h, (rmax_256 + CROP_MARGIN) * scale_r))
    cmin = int(max(0, (cmin_256 - CROP_MARGIN) * scale_c))
    cmax = int(min(original_w, (cmax_256 + CROP_MARGIN) * scale_c))

    return stage1_pred, (rmin, rmax, cmin, cmax)


# -------------------------------------------------------------------------
# Stage 2: Fine-grained tumor detection on crop
# -------------------------------------------------------------------------
def run_stage2(image_np, bbox, model_s2, device):
    """
    Crop the original image to the bounding box and run Stage 2.
    Returns:
        stage2_pred_crop: (128, 128) prediction on the resized crop
        crop: the actual cropped image region (numpy), original size
    """
    rmin, rmax, cmin, cmax = bbox

    # Crop the original full-resolution image to the pancreas region
    crop = image_np[rmin:rmax, cmin:cmax]

    # Resize crop to 128x128 for Stage 2
    tensor = torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    tensor_128 = F.interpolate(tensor, size=(STAGE2_INPUT_SIZE, STAGE2_INPUT_SIZE),
                                mode='bilinear', align_corners=False).to(device)

    with torch.no_grad():
        output = model_s2(tensor_128)              # (1, 3, 128, 128)
        probs = torch.softmax(output, dim=1)       # (1, 3, 128, 128)
        pred = torch.argmax(output, dim=1)          # (1, 128, 128)

    stage2_pred_crop = pred[0].cpu().numpy()       # (128, 128), values 0, 1, 2
    stage2_probs_crop = probs[0].cpu().numpy()     # (3, 128, 128)

    return stage2_pred_crop, stage2_probs_crop, crop


# -------------------------------------------------------------------------
# Combine: map Stage 2 crop prediction back to full-slice coordinates
# -------------------------------------------------------------------------
def combine_predictions(image_np, stage1_pred, stage2_pred_crop, bbox):
    """
    Place the Stage 2 crop prediction back into the original full-slice canvas.
    Returns full_pred: same shape as image_np with classes 0, 1, 2.
    """
    original_h, original_w = image_np.shape
    rmin, rmax, cmin, cmax = bbox

    crop_h = rmax - rmin
    crop_w = cmax - cmin

    # Resize Stage 2 prediction from (128,128) back to original crop dimensions
    pred_tensor = torch.from_numpy(stage2_pred_crop).float().unsqueeze(0).unsqueeze(0)
    pred_resized = F.interpolate(pred_tensor, size=(crop_h, crop_w),
                                  mode='nearest')
    pred_resized = pred_resized.squeeze().numpy().astype(int)

    # Start with all background
    full_pred = np.zeros((original_h, original_w), dtype=int)

    # Place the Stage 2 result into the bounding box region
    full_pred[rmin:rmax, cmin:cmax] = pred_resized

    return full_pred


# -------------------------------------------------------------------------
# Full pipeline: image → final prediction
# -------------------------------------------------------------------------
def run_pipeline(image_np, model_s1, model_s2, device):
    """
    Run the complete two-stage pipeline on a single CT slice.

    Args:
        image_np: 2D numpy array, CT slice (values 0-1, already windowed)
        model_s1: loaded Stage 1 model
        model_s2: loaded Stage 2 model
        device: torch device

    Returns:
        full_pred: full-size prediction map (0=bg, 1=pancreas, 2=tumor)
        stage1_pred: Stage 1 binary mask (256x256)
        crop: the cropped image region, or None if no pancreas found
        bbox: (rmin, rmax, cmin, cmax) or None
        stats: dict with detection info and confidence scores
    """
    # Stage 1
    stage1_pred, bbox = run_stage1(image_np, model_s1, device)

    if bbox is None:
        # No pancreas found by Stage 1 — return all-background prediction
        full_pred = np.zeros(image_np.shape, dtype=int)
        stats = {
            'pancreas_detected': False,
            'tumor_detected': False,
            'max_tumor_confidence': 0.0,
            'background_pct': 100.0,
            'pancreas_pct': 0.0,
            'tumor_pct': 0.0,
        }
        return full_pred, stage1_pred, None, None, stats

    # Stage 2
    stage2_pred_crop, stage2_probs_crop, crop = run_stage2(image_np, bbox, model_s2, device)

    # Map Stage 2 result back to full-slice canvas
    full_pred = combine_predictions(image_np, stage1_pred, stage2_pred_crop, bbox)

    total = full_pred.size
    stats = {
        'pancreas_detected': bool(np.any(full_pred == 1)),
        'tumor_detected': bool(np.any(full_pred == 2)),
        'max_tumor_confidence': float(np.max(stage2_probs_crop[2])),
        'background_pct': float(np.sum(full_pred == 0) / total * 100),
        'pancreas_pct': float(np.sum(full_pred == 1) / total * 100),
        'tumor_pct': float(np.sum(full_pred == 2) / total * 100),
    }

    return full_pred, stage1_pred, crop, bbox, stats


# -------------------------------------------------------------------------
# Visualization helper
# -------------------------------------------------------------------------
def visualize_pipeline(image_np, stage1_pred_256, crop, bbox, full_pred, stats, save_path=None):
    """Create a 4-panel visualization of the full two-stage inference."""
    n_cols = 4
    fig, axes = plt.subplots(1, n_cols, figsize=(20, 5))
    fig.patch.set_facecolor('#060a14')

    for ax in axes:
        ax.set_facecolor('#0c1221')
        ax.axis('off')

    # Panel 1: Original CT slice
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title('Original CT Slice', color='#e8ecf4', pad=10)

    # Panel 2: Stage 1 — pancreas localization (256x256 binary mask)
    axes[1].imshow(np.zeros((STAGE1_INPUT_SIZE, STAGE1_INPUT_SIZE)), cmap='gray')
    s1_overlay = np.zeros((*stage1_pred_256.shape, 4), dtype=np.float32)
    s1_overlay[stage1_pred_256 == 1] = [0.2, 0.83, 0.67, 0.6]  # teal for pancreas region
    axes[1].imshow(s1_overlay)
    if bbox is not None:
        # Draw the bounding box on a resized version
        h, w = image_np.shape
        scale_r = STAGE1_INPUT_SIZE / h
        scale_c = STAGE1_INPUT_SIZE / w
        rmin, rmax, cmin, cmax = bbox
        r0, r1 = int(rmin * scale_r), int(rmax * scale_r)
        c0, c1 = int(cmin * scale_c), int(cmax * scale_c)
        from matplotlib.patches import Rectangle
        rect = Rectangle((c0, r0), c1 - c0, r1 - r0,
                          linewidth=2, edgecolor='#ffb347', facecolor='none')
        axes[1].add_patch(rect)
    axes[1].set_title('Stage 1: Pancreas Localization', color='#e8ecf4', pad=10)

    # Panel 3: Stage 2 — crop close-up
    if crop is not None:
        axes[2].imshow(crop, cmap='gray')
        axes[2].set_title('Stage 2: Zoomed Crop', color='#e8ecf4', pad=10)
    else:
        axes[2].set_title('No pancreas detected', color='#ff4757', pad=10)

    # Panel 4: Final combined prediction on full slice
    axes[3].imshow(image_np, cmap='gray')
    overlay = np.zeros((*full_pred.shape, 4), dtype=np.float32)
    overlay[full_pred == 1] = [0.2, 0.83, 0.67, 0.55]   # teal = pancreas
    overlay[full_pred == 2] = [1.0, 0.28, 0.34, 0.65]   # red = tumor
    axes[3].imshow(overlay)

    status = "🔴 TUMOR DETECTED" if stats['tumor_detected'] else "✅ No Tumor" if stats['pancreas_detected'] else "⬛ No Pancreas"
    conf = f" (conf: {stats['max_tumor_confidence']:.2%})" if stats['tumor_detected'] else ""
    axes[3].set_title(f'Final Prediction\n{status}{conf}', color='#e8ecf4', pad=10)

    plt.tight_layout(pad=2)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved visualization to {save_path}")

    plt.show()
    plt.close(fig)


# -------------------------------------------------------------------------
# CLI: Run pipeline on a sample from the validation set
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check both models exist before starting
    if not os.path.exists(STAGE1_MODEL_PATH):
        print(f"\nERROR: Stage 1 model not found at {STAGE1_MODEL_PATH}")
        print("Run `python train_stage1.py` first to train Stage 1.")
        sys.exit(1)

    if not os.path.exists(STAGE2_MODEL_PATH):
        print(f"\nERROR: Stage 2 model not found at {STAGE2_MODEL_PATH}")
        print("Run `python train_stage2.py` first to train Stage 2.")
        sys.exit(1)

    model_s1, model_s2 = load_models(device)

    # Load a validation slice that contains a tumor for a meaningful demo
    split_file = '../data/processed/split.json'
    images_dir = '../data/processed/images'
    labels_dir = '../data/processed/labels'

    with open(split_file) as f:
        val_files = json.load(f)['val']

    # Find a val slice that contains cancer (class 2) for the best demo
    test_file = None
    print("\nSearching for a validation slice with tumor annotation...")
    for fname in val_files:
        label = np.load(f"{labels_dir}/{fname}")
        if np.any(label == 2):
            test_file = fname
            break

    if test_file is None:
        print("No tumor slice found, using first val slice instead.")
        test_file = val_files[0]

    print(f"Running pipeline on: {test_file}")

    image_np = np.load(f"{images_dir}/{test_file}")
    full_pred, s1_pred, crop, bbox, stats = run_pipeline(image_np, model_s1, model_s2, device)

    print("\n--- Pipeline Results ---")
    print(f"Pancreas detected: {stats['pancreas_detected']}")
    print(f"Tumor detected:    {stats['tumor_detected']}")
    print(f"Max tumor confidence: {stats['max_tumor_confidence']:.4f}")
    print(f"Background: {stats['background_pct']:.1f}%  "
          f"Pancreas: {stats['pancreas_pct']:.1f}%  "
          f"Tumor: {stats['tumor_pct']:.1f}%")

    os.makedirs('../outputs', exist_ok=True)
    visualize_pipeline(image_np, s1_pred, crop, bbox, full_pred, stats,
                       save_path='../outputs/pipeline_demo.png')
