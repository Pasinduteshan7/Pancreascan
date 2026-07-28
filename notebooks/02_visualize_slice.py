import sys
sys.path.append('../src')

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from config import DATA_ROOT

img = nib.load(f'{DATA_ROOT}/imagesTr/pancreas_001.nii.gz')
lbl = nib.load(f'{DATA_ROOT}/labelsTr/pancreas_001.nii.gz')

image_data = img.get_fdata()
label_data = lbl.get_fdata()

# Step 1: find which slices actually contain cancer (label == 2)
cancer_slices = np.where(np.any(label_data == 2, axis=(0, 1)))[0]
print("Slices containing cancer:", cancer_slices)

# Step 2: pick one of those slices to look at
slice_idx = cancer_slices[len(cancer_slices) // 2]  # a middle one from the list
print("Showing slice:", slice_idx)

ct_slice = image_data[:, :, slice_idx]
label_slice = label_data[:, :, slice_idx]

# Step 3: plot side by side — raw CT vs. CT with tumor overlay
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(ct_slice, cmap='gray')
axes[0].set_title(f'Raw CT slice {slice_idx}')
axes[0].axis('off')

axes[1].imshow(ct_slice, cmap='gray')
# Mask out background so only pancreas/cancer show as colored overlay
masked_label = np.ma.masked_where(label_slice == 0, label_slice)
axes[1].imshow(masked_label, cmap='autumn', alpha=0.5)
axes[1].set_title(f'Slice {slice_idx} with pancreas/tumor overlay')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('../outputs/sample_slice_overlay.png', dpi=150)
plt.show()