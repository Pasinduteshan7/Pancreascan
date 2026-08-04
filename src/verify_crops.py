# src/verify_crops.py
import os
import numpy as np
import matplotlib.pyplot as plt

CROPPED_DIR = "../data/processed_cropped"

def verify_crops(num_examples=6):
    image_files = sorted(os.listdir(f"{CROPPED_DIR}/images"))

    # Pick a spread of examples, not just the first few
    step = max(1, len(image_files) // num_examples)
    sample_files = image_files[::step][:num_examples]

    fig, axes = plt.subplots(num_examples, 2, figsize=(8, 4 * num_examples))

    for i, filename in enumerate(sample_files):
        image = np.load(f"{CROPPED_DIR}/images/{filename}")
        label = np.load(f"{CROPPED_DIR}/labels/{filename}")

        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f'{filename}\nshape={image.shape}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(image, cmap='gray')
        masked_label = np.ma.masked_where(label == 0, label)
        axes[i, 1].imshow(masked_label, cmap='autumn', alpha=0.5, vmin=0, vmax=2)
        axes[i, 1].set_title('Cropped + overlay')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('../outputs/crop_verification.png', dpi=150)
    plt.show()

    # Also print some basic stats across ALL files, not just the visual sample
    print("\n--- Checking all cropped files for basic sanity ---")
    shapes = []
    empty_labels = 0

    for filename in image_files:
        image = np.load(f"{CROPPED_DIR}/images/{filename}")
        label = np.load(f"{CROPPED_DIR}/labels/{filename}")

        if image.shape != label.shape:
            print(f"MISMATCH shape: {filename} image={image.shape} label={label.shape}")

        if not np.any(label > 0):
            empty_labels += 1

        shapes.append(image.shape)

    heights = [s[0] for s in shapes]
    widths = [s[1] for s in shapes]
    print(f"Total files: {len(image_files)}")
    print(f"Empty labels (no pancreas/cancer after crop): {empty_labels}")
    print(f"Height range: {min(heights)} to {max(heights)}")
    print(f"Width range: {min(widths)} to {max(widths)}")

if __name__ == "__main__":
    verify_crops()