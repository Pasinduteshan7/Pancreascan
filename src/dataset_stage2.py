# src/dataset_stage2.py
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Default directories to search for cropped slice files (checked in order)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_IMAGE_DIRS = [
    os.path.join(PROJECT_DIR, "data", "processed_cropped", "images"),
    os.path.join(PROJECT_DIR, "data", "panorama_cropped", "images"),
]
DEFAULT_LABEL_DIRS = [
    os.path.join(PROJECT_DIR, "data", "processed_cropped", "labels"),
    os.path.join(PROJECT_DIR, "data", "panorama_cropped", "labels"),
]


class Stage2Dataset(Dataset):
    """
    3-class segmentation dataset for Stage 2 (fine-grained tumor detection).
    Uses pre-cropped data — tightly cropped around the pancreas region,
    so the model focuses only on pancreas tissue.
    Labels: 0 = background (within crop), 1 = healthy pancreas, 2 = tumor.

    Supports loading files from multiple directories (e.g., MSD + PANORAMA).
    Files are searched in each directory in order until found.
    """
    def __init__(self, split='train', split_file=None,
                 images_dir=None, labels_dir=None,
                 images_dirs=None, labels_dirs=None,
                 target_size=128):  # smaller than Stage 1 — crops are already zoomed in
        # Determine split file path
        if split_file is None:
            split_file = os.path.join(PROJECT_DIR, "data", "processed", "split.json")

        with open(split_file) as f:
            splits = json.load(f)

        all_filenames = splits[split]
        self.target_size = target_size

        # Support both single-dir (backward compat) and multi-dir modes
        if images_dirs is not None:
            self.images_dirs = images_dirs
        elif images_dir is not None:
            self.images_dirs = [images_dir]
        else:
            self.images_dirs = DEFAULT_IMAGE_DIRS

        if labels_dirs is not None:
            self.labels_dirs = labels_dirs
        elif labels_dir is not None:
            self.labels_dirs = [labels_dir]
        else:
            self.labels_dirs = DEFAULT_LABEL_DIRS

        # Build a lookup cache: filename -> (image_path, label_path)
        # Only include files that actually exist in at least one directory
        self._path_cache = {}
        self.filenames = []
        for fname in all_filenames:
            img_path = self._find_file(fname, self.images_dirs)
            lbl_path = self._find_file(fname, self.labels_dirs)
            if img_path and lbl_path:
                self._path_cache[fname] = (img_path, lbl_path)
                self.filenames.append(fname)

    def _find_file(self, filename, dirs):
        """Search for a file across multiple directories."""
        for d in dirs:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path, lbl_path = self._path_cache[filename]

        image = np.load(img_path)
        label = np.load(lbl_path)

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

