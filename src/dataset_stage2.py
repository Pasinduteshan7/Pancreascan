# src/dataset_stage2.py
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Stage2Dataset(Dataset):
    """
    3-class segmentation dataset for Stage 2 (fine-grained tumor detection).
    Uses the pre-cropped data from data/processed_cropped/ — tightly cropped
    around the pancreas region, so the model focuses only on pancreas tissue.
    Labels: 0 = background (within crop), 1 = healthy pancreas, 2 = tumor.
    """
    def __init__(self, split='train', split_file='../data/processed/split.json',
                 images_dir='../data/processed_cropped/images',
                 labels_dir='../data/processed_cropped/labels',
                 target_size=128):  # smaller than Stage 1 — crops are already zoomed in
        with open(split_file) as f:
            splits = json.load(f)

        # Filter to only files that actually exist in the cropped directory
        all_filenames = splits[split]
        self.filenames = [f for f in all_filenames
                          if __import__('os').path.exists(f"{images_dir}/{f}")]

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = np.load(f"{self.images_dir}/{filename}")
        label = np.load(f"{self.labels_dir}/{filename}")

        # Crops are variable size — resize to fixed target_size for batching
        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        label = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0)   # (1,1,H,W)

        image = F.interpolate(image, size=(self.target_size, self.target_size),
                              mode='bilinear', align_corners=False)
        label = F.interpolate(label, size=(self.target_size, self.target_size),
                              mode='nearest')

        image = image.squeeze(0)                            # (1, H, W)
        label = label.squeeze(0).squeeze(0).long()          # (H, W), values 0, 1, 2

        return image, label


if __name__ == "__main__":
    train_ds = Stage2Dataset(split='train')
    val_ds = Stage2Dataset(split='val')

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    image, label = train_ds[0]
    print(f"Image shape:  {image.shape}, dtype: {image.dtype}")
    print(f"Label shape:  {label.shape}, dtype: {label.dtype}")
    print(f"Unique label values: {torch.unique(label)}")  # Should include 0, 1, and ideally 2
