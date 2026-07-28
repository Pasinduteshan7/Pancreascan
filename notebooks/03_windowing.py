import sys
sys.path.append('../src')

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from config import DATA_ROOT
from preprocessing import apply_window

img = nib.load(f'{DATA_ROOT}/imagesTr/pancreas_001.nii.gz')
lbl = nib.load(f'{DATA_ROOT}/labelsTr/pancreas_001.nii.gz')

image_data = img.get_fdata()
label_data = lbl.get_fdata()

slice_idx = 44  # same slice as before, so we can compare directly
ct_slice = image_data[:, :, slice_idx]
label_slice = label_data[:, :, slice_idx]

windowed_slice = apply_window(ct_slice)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

axes[0].imshow(ct_slice, cmap='gray')
axes[0].set_title('Raw CT (unwindowed)')
axes[0].axis('off')

axes[1].imshow(windowed_slice, cmap='gray')
axes[1].set_title('Windowed (soft tissue)')
axes[1].axis('off')

axes[2].imshow(windowed_slice, cmap='gray')
masked_label = np.ma.masked_where(label_slice == 0, label_slice)
axes[2].imshow(masked_label, cmap='autumn', alpha=0.5)
axes[2].set_title('Windowed + tumor overlay')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('../outputs/windowing_comparison.png', dpi=150)
plt.show()

print("Raw slice range:", ct_slice.min(), "to", ct_slice.max())
print("Windowed slice range:", windowed_slice.min(), "to", windowed_slice.max())