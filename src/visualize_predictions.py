# src/visualize_predictions.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import PancreasDataset
from model import UNet

def visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_ds = PancreasDataset(split='val')
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=0)

    model = UNet(in_channels=1, num_classes=3).to(device)
    model.load_state_dict(torch.load('../outputs/best_model.pth', map_location=device, weights_only=True))
    model.eval()

    num_examples = 6
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4 * num_examples))

    shown = 0
    with torch.no_grad():
        for images, labels in val_loader:
            # Only show slices that actually contain cancer - more informative
            if 2 not in labels.unique():
                continue

            images_gpu = images.to(device)
            outputs = model(images_gpu)
            preds = torch.argmax(outputs, dim=1).cpu()

            img = images[0, 0].numpy()          # (H, W)
            true_label = labels[0].numpy()       # (H, W)
            pred_label = preds[0].numpy()        # (H, W)

            # Column 1: raw CT slice
            axes[shown, 0].imshow(img, cmap='gray')
            axes[shown, 0].set_title('CT Slice')
            axes[shown, 0].axis('off')

            # Column 2: ground truth overlay
            axes[shown, 1].imshow(img, cmap='gray')
            masked_true = np.ma.masked_where(true_label == 0, true_label)
            axes[shown, 1].imshow(masked_true, cmap='autumn', alpha=0.5, vmin=0, vmax=2)
            axes[shown, 1].set_title('Ground Truth')
            axes[shown, 1].axis('off')

            # Column 3: model prediction overlay
            axes[shown, 2].imshow(img, cmap='gray')
            masked_pred = np.ma.masked_where(pred_label == 0, pred_label)
            axes[shown, 2].imshow(masked_pred, cmap='autumn', alpha=0.5, vmin=0, vmax=2)
            axes[shown, 2].set_title('Model Prediction')
            axes[shown, 2].axis('off')

            shown += 1
            if shown >= num_examples:
                break

    plt.tight_layout()
    plt.savefig('../outputs/predictions_comparison.png', dpi=150)
    plt.show()
    print(f"Saved comparison to ../outputs/predictions_comparison.png")

if __name__ == "__main__":
    visualize()