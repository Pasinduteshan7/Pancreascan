# src/evaluate.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PancreasDataset
from model import UNet

def dice_score(pred, target, class_id, epsilon=1e-6):
    """
    Compute Dice score for one class.
    pred, target: tensors of shape (H, W) with class indices (0,1,2)
    """
    pred_mask = (pred == class_id).float()
    target_mask = (target == class_id).float()

    intersection = (pred_mask * target_mask).sum()
    total = pred_mask.sum() + target_mask.sum()

    # epsilon avoids division by zero if a class is absent in both
    dice = (2 * intersection + epsilon) / (total + epsilon)
    return dice.item()


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    val_ds = PancreasDataset(split='val')
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = UNet(in_channels=1, num_classes=3).to(device)
    model.load_state_dict(torch.load('../outputs/best_model.pth', map_location=device, weights_only=True))
    model.eval()

    class_names = {0: 'background', 1: 'pancreas', 2: 'cancer'}
    dice_totals = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)                    # (B, 3, H, W) raw scores
            preds = torch.argmax(outputs, dim=1)        # (B, H, W) predicted class per pixel

            for i in range(images.shape[0]):
                for class_id in [0, 1, 2]:
                    d = dice_score(preds[i], labels[i], class_id)
                    dice_totals[class_id].append(d)

    print("\n--- Validation Dice Scores ---")
    for class_id in [0, 1, 2]:
        scores = dice_totals[class_id]
        mean_dice = np.mean(scores)
        print(f"{class_names[class_id]:12s}: mean Dice = {mean_dice:.4f}")

if __name__ == "__main__":
    evaluate()