# src/visualize_pipeline_predictions.py
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import PancreasDataset          # original full-slice dataset (for ground truth + Stage 1 input)
from model import UNet

# Resolve paths safely regardless of current working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

STAGE1_PATH = os.path.join(OUTPUTS_DIR, "best_model_stage1.pth")
STAGE2_PATH = os.path.join(OUTPUTS_DIR, "best_model_stage2.pth")
OUTPUT_IMG_PATH = os.path.join(OUTPUTS_DIR, "two_stage_verification.png")

def get_bounding_box(mask, margin=20):
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = mask.shape
    return (max(0, rmin-margin), min(h, rmax+margin),
            max(0, cmin-margin), min(w, cmax+margin))

def visualize(num_examples=6, split_file=None, seed=42):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_kwargs = {'split': 'val'}
    if split_file is not None:
        dataset_kwargs['split_file'] = split_file
    elif os.path.exists(os.path.join(PROJECT_DIR, "data", "processed", "split_combined.json")):
        dataset_kwargs['split_file'] = os.path.join(PROJECT_DIR, "data", "processed", "split_combined.json")

    val_ds = PancreasDataset(**dataset_kwargs)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=0)

    stage1 = UNet(in_channels=1, num_classes=2).to(device)
    s1_path = STAGE1_PATH if os.path.exists(STAGE1_PATH) else '../outputs/best_model_stage1.pth'
    stage1.load_state_dict(torch.load(s1_path, map_location=device, weights_only=True))
    stage1.eval()

    stage2 = UNet(in_channels=1, num_classes=3).to(device)
    s2_path = STAGE2_PATH if os.path.exists(STAGE2_PATH) else '../outputs/best_model_stage2.pth'
    stage2.load_state_dict(torch.load(s2_path, map_location=device, weights_only=True))
    stage2.eval()

    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4 * num_examples))
    if num_examples == 1:
        axes = np.expand_dims(axes, 0)
    shown = 0

    with torch.no_grad():
        for images, labels in val_loader:
            if 2 not in labels.unique():
                continue  # only show slices with real tumor present

            img_np = images[0, 0].numpy()
            true_label = labels[0].numpy()

            # Stage 1: localize
            s1_out = stage1(images.to(device))
            s1_pred = torch.argmax(s1_out, dim=1)[0].cpu().numpy()

            bbox = get_bounding_box(s1_pred)
            if bbox is None:
                continue
            rmin, rmax, cmin, cmax = bbox

            # Crop for Stage 2
            crop = img_np[rmin:rmax, cmin:cmax]
            crop_tensor = torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0)
            crop_tensor = torch.nn.functional.interpolate(crop_tensor, size=(128,128), mode='bilinear', align_corners=False).to(device)

            s2_out = stage2(crop_tensor)
            s2_pred = torch.argmax(s2_out, dim=1)[0].cpu().numpy()

            # Map Stage 2 prediction back onto full-size canvas for display
            full_pred = np.zeros_like(true_label)
            s2_resized = torch.nn.functional.interpolate(
                torch.from_numpy(s2_pred).float().unsqueeze(0).unsqueeze(0),
                size=(rmax-rmin, cmax-cmin), mode='nearest'
            )[0,0].numpy()
            full_pred[rmin:rmax, cmin:cmax] = s2_resized

            # Column 1: raw CT
            axes[shown,0].imshow(img_np, cmap='gray'); axes[shown,0].set_title('CT Slice'); axes[shown,0].axis('off')
            # Column 2: Stage 1 mask
            axes[shown,1].imshow(img_np, cmap='gray')
            axes[shown,1].imshow(np.ma.masked_where(s1_pred==0, s1_pred), cmap='cool', alpha=0.5)
            axes[shown,1].set_title('Stage 1: Localization'); axes[shown,1].axis('off')
            # Column 3: Ground truth
            axes[shown,2].imshow(img_np, cmap='gray')
            axes[shown,2].imshow(np.ma.masked_where(true_label==0, true_label), cmap='autumn', alpha=0.5, vmin=0, vmax=2)
            axes[shown,2].set_title('Ground Truth'); axes[shown,2].axis('off')
            # Column 4: Final two-stage prediction
            axes[shown,3].imshow(img_np, cmap='gray')
            axes[shown,3].imshow(np.ma.masked_where(full_pred==0, full_pred), cmap='autumn', alpha=0.5, vmin=0, vmax=2)
            axes[shown,3].set_title('Two-Stage Prediction'); axes[shown,3].axis('off')

            shown += 1
            if shown >= num_examples:
                break

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=150)
    print(f"Saved {shown} examples to {OUTPUT_IMG_PATH}")
    try:
        if sys.stdout.isatty() and not os.environ.get("NON_INTERACTIVE"):
            plt.show(block=False)
            plt.pause(0.5)
    except Exception:
        pass
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize two-stage pipeline predictions")
    parser.add_argument("--num-examples", type=int, default=6, help="Number of tumor examples to visualize")
    parser.add_argument("--split-file", type=str, default=None, help="Path to split.json file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    visualize(num_examples=args.num_examples, split_file=args.split_file, seed=args.seed)
