# src/dataset.py
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PancreasDataset(Dataset):
    def __init__(self, split='train', split_file='../data/processed/split.json',
                 images_dir='../data/processed/images', labels_dir='../data/processed/labels'):
        with open(split_file) as f:
            splits = json.load(f)

        self.filenames = splits[split]
        self.images_dir = images_dir
        self.labels_dir = labels_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = np.load(f"{self.images_dir}/{filename}")
        label = np.load(f"{self.labels_dir}/{filename}")

        # Add channel dimension: (H, W) -> (1, H, W)
        image = torch.from_numpy(image).float().unsqueeze(0)
        label = torch.from_numpy(label).long()

        return image, label


if __name__ == "__main__":
    train_ds = PancreasDataset(split='train')
    val_ds = PancreasDataset(split='val')

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    # Check a single sample first
    image, label = train_ds[0]
    print(f"Single image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Single label shape: {label.shape}, dtype: {label.dtype}")
    print(f"Unique label values: {torch.unique(label)}")

    # Now check batching via DataLoader
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")   # expect [4, 1, 512, 512]
    print(f"Batch label shape: {labels.shape}")   # expect [4, 512, 512]