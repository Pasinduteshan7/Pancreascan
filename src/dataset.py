# src/dataset.py
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class PancreasDataset(Dataset):
    def __init__(self, split='train', split_file='../data/processed/split.json',
                 images_dir='../data/processed/images', labels_dir='../data/processed/labels',
                 target_size=256):
        with open(split_file) as f:
            splits = json.load(f)

        self.filenames = splits[split]
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = np.load(f"{self.images_dir}/{filename}")
        label = np.load(f"{self.labels_dir}/{filename}")

        # Convert to tensors, add channel + batch dims temporarily for resizing
        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        label = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # Resize to smaller resolution for speed
        image = F.interpolate(image, size=(self.target_size, self.target_size),
                               mode='bilinear', align_corners=False)
        label = F.interpolate(label, size=(self.target_size, self.target_size),
                               mode='nearest')

        # Remove the extra batch dim, keep channel dim for image; labels have no channel dim
        image = image.squeeze(0)          # (1, H, W)
        label = label.squeeze(0).squeeze(0).long()  # (H, W)

        return image, label


if __name__ == "__main__":
    train_ds = PancreasDataset(split='train')
    val_ds = PancreasDataset(split='val')

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    image, label = train_ds[0]
    print(f"Single image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Single label shape: {label.shape}, dtype: {label.dtype}")
    print(f"Unique label values: {torch.unique(label)}")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")