# src/predict_confidence.py
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PancreasDataset
from model import UNet

def compute_confidence_scores():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_ds = PancreasDataset(split='val')
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = UNet(in_channels=1, num_classes=3).to(device)
    model.load_state_dict(torch.load('../outputs/best_model.pth', map_location=device, weights_only=True))
    model.eval()

    results = {}

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            filename = val_ds.filenames[idx]
            images = images.to(device)

            outputs = model(images)                        # (1, 3, H, W) raw scores
            probs = F.softmax(outputs, dim=1)               # convert to probabilities (0-1, sum to 1 per pixel)

            cancer_prob_map = probs[0, 2]                   # probability of "cancer" class at every pixel
            max_cancer_prob = cancer_prob_map.max().item()  # highest confidence anywhere in this slice

            results[filename] = round(max_cancer_prob, 4)

    with open('../outputs/confidence_scores.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved confidence scores for {len(results)} slices to ../outputs/confidence_scores.json")

    # Show a few examples
    print("\nSample scores:")
    for filename, score in list(results.items())[:5]:
        print(f"  {filename}: {score}")

if __name__ == "__main__":
    compute_confidence_scores()