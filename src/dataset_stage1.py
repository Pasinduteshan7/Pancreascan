# src/dataset_stage1.py
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Stage1Dataset(Dataset):
    """
    Binary segmentation dataset for Stage 1 (pancreas localization).
    Labels: 0 = background, 1 = pancreas-region (pancreas + cancer merged into one class).
    Stage 1's job is just: WHERE is the pancreas — not what type of tissue it is.
    """
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

        # Merge class 1 (pancreas) and class 2 (cancer) → both become class 1
        # Result: 0=background, 1=pancreas-region (includes cancer inside pancreas)
        binary_label = (label > 0).astype(np.float32)

        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)           # (1,1,H,W)
        binary_label = torch.from_numpy(binary_label).unsqueeze(0).unsqueeze(0)     # (1,1,H,W)

        image = F.interpolate(image, size=(self.target_size, self.target_size),
                              mode='bilinear', align_corners=False)
        binary_label = F.interpolate(binary_label, size=(self.target_size, self.target_size),
                                     mode='nearest')

        image = image.squeeze(0)                               # (1, H, W)
        binary_label = binary_label.squeeze(0).squeeze(0).long()  # (H, W), values 0 or 1

        return image, binary_label


if __name__ == "__main__":
    train_ds = Stage1Dataset(split='train')
    val_ds = Stage1Dataset(split='val')

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    image, label = train_ds[0]
    print(f"Image shape:  {image.shape}, dtype: {image.dtype}")
    print(f"Label shape:  {label.shape}, dtype: {label.dtype}")
    print(f"Unique label values: {torch.unique(label)}")  # Should be [0, 1] only
