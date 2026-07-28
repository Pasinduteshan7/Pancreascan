import sys
sys.path.append('../src')  # so we can import config.py

import nibabel as nib
import numpy as np
from config import DATA_ROOT

img = nib.load(f'{DATA_ROOT}/imagesTr/pancreas_001.nii.gz')
lbl = nib.load(f'{DATA_ROOT}/labelsTr/pancreas_001.nii.gz')

image_data = img.get_fdata()
label_data = lbl.get_fdata()

print("Image shape:", image_data.shape)
print("Label shape:", label_data.shape)
print("Voxel spacing:", img.header.get_zooms())
print("HU value range:", image_data.min(), "to", image_data.max())
print("Unique label values:", np.unique(label_data))