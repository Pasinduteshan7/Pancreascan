# src/dataset_stage1.py
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Default directories to search for slice files (checked in order)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_IMAGE_DIRS = [
    os.path.join(PROJECT_DIR, "data", "processed", "images"),
    os.path.join(PROJECT_DIR, "data", "panorama_processed", "images"),
]
DEFAULT_LABEL_DIRS = [
    os.path.join(PROJECT_DIR, "data", "processed", "labels"),
    os.path.join(PROJECT_DIR, "data", "panorama_processed", "labels"),
]


class Stage1Dataset(Dataset):
    """
    Binary segmentation dataset for Stage 1 (pancreas localization).
    Labels: 0 = background, 1 = pancreas-region (pancreas + cancer merged into one class).
    Stage 1's job is just: WHERE is the pancreas — not what type of tissue it is.

    Supports loading files from multiple directories (e.g., MSD + PANORAMA).
    Files are searched in each directory in order until found.
    """
    def __init__(self, split='train', split_file=None,
                 images_dir=None, labels_dir=None,
                 images_dirs=None, labels_dirs=None,
                 target_size=256):
        # Determine split file path
        if split_file is None:
            split_file = os.path.join(PROJECT_DIR, "data", "processed", "split.json")

        with open(split_file) as f:
            splits = json.load(f)

        self.filenames = splits[split]
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
        self._path_cache = {}
        valid_files = []
        for fname in self.filenames:
            img_path = self._find_file(fname, self.images_dirs)
            lbl_path = self._find_file(fname, self.labels_dirs)
            if img_path and lbl_path:
                self._path_cache[fname] = (img_path, lbl_path)
                valid_files.append(fname)

        self.filenames = valid_files

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

