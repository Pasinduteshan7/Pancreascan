# PancreaScan

An AI-based system for detecting pancreatic tumors from CT imaging. This project 
processes volumetric CT data (NIfTI format) from the Medical Segmentation Decathlon 
(Task07_Pancreas) dataset, converts 3D volumes into 2D slices, and applies deep 
learning to detect and segment pancreatic tumors.

## Project Structure

project/
├── data/ # dataset (not committed - see setup below)
├── notebooks/ # exploration and visualization scripts
├── src/ # core pipeline code
│ ├── config.py # dataset path configuration
│ ├── preprocessing.py # CT windowing/normalization
│ ├── extract_slices.py # 3D -> 2D slice extraction
│ └── create_splits.py # patient-level train/val split
├── outputs/ # generated images, results
└── requirements.txt


## Setup

1. Download the Medical Segmentation Decathlon Task07_Pancreas dataset:
   http://medicaldecathlon.com/

2. Update `src/config.py` with your local dataset path.

3. Install dependencies:

pip install -r requirements.txt


4. Run the pipeline:

cd src
python extract_slices.py
python create_splits.py


## Dataset
Medical Segmentation Decathlon — Task07_Pancreas  
281 CT scans with pancreas + tumor segmentation labels (CC-BY-SA 4.0)

## Data Processing Concepts

If you are new to medical imaging or computer vision, here is a breakdown of the core concepts underlying our data pipeline:

### 1. What is a digital image, fundamentally?
A regular photo has 3 numbers per pixel (Red, Green, Blue — RGB), each 0-255. A grayscale image has just 1 number per pixel — its brightness.
Our CT scan is a **3D grid of numbers** — a grayscale value at every point in a 3D volume, not just a flat 2D grid. All "image processing" is really just math operations on arrays of numbers.

### 2. Loading the NIfTI file
```python
img = nib.load(...)
volume = img.get_fdata()
```
- **Voxels vs pixels**: In a 3D volume, the smallest unit is a voxel (volume element) — it represents a tiny cube of real physical space inside the patient's body, holding one number: how dense that tissue is.
- **Hounsfield Units (HU)**: CT scanners measure physical density calibrated to a scale where water = 0, air = -1000, bone = 1000+. These numbers have real physical meaning.

### 3. Extracting a 2D slice
```python
ct_slice = image_data[:, :, slice_idx]
```
- **Array slicing**: We select one 2D layer out of a 3D array (like pulling one page out of a stack). This gives a flat 512×512 grid of HU values behaving like an ordinary grayscale image.

### 4. Windowing
```python
def apply_window(slice_2d, center=40, width=400):
    low, high = center - width//2, center + width//2
    windowed = np.clip(slice_2d, low, high)
    normalized = (windowed - low) / (high - low)
```
- **Contrast/intensity mapping**: Raw data spans ~4000 possible values, but soft tissue is only a tiny sliver (-160 to +240). Displaying the whole range compresses soft tissue into near-identical gray shades.
- **Clipping**: `np.clip()` discards information outside our area of interest, thresholding the image to isolate the relevant signal (organs).
- **Normalization**: Rescaling the remaining range to `0.0–1.0`. Neural networks train far better on small, consistent numeric ranges.

### 5. Finding useful slices
```python
useful_slices = np.where(np.any(label_data > 0, axis=(0, 1)))[0]
```
- **Masks and boolean arrays**: `label_data > 0` creates a binary mask (True/False values indicating the presence of pancreas or cancer). 
- **Reduction**: We collapse dimensions to intelligently select which slices actually contain anatomy worth training on, ignoring empty slices.

### 6. Building the Dataset class
```python
image = torch.from_numpy(image).float().unsqueeze(0)
```
- **Tensors**: PyTorch's multidimensional arrays with GPU support.
- **Channels**: Computer vision requires a channel dimension `(channels, height, width)`. `unsqueeze(0)` explicitly adds a dimension of size 1 for our grayscale slice to satisfy PyTorch's convolution layers.

### 7. Batching
Batching stacks multiple independent samples together into one larger array `(batch, channels, height, width)`. This allows the GPU to process several images simultaneously for hardware efficiency.

### The Full Transformation Summary
1. **Raw CT scan** (3D array, HU values -1024 to 3071)
2. **2D slice** via array indexing (512×512, still raw HU)
3. **Windowing** via clip + normalize (512×512, values 0.0-1.0, soft tissue contrast enhanced)
4. **Filter by mask** to keep only slices with anatomy present
5. **Tensor conversion** and channel dimension addition → `(1, 512, 512)`
6. **Batching** via DataLoader → `(4, 1, 512, 512)` ready for the neural network.

## Status
🚧 In progress — data pipeline complete, model training next.

## License
CC-BY-SA 4.0 (dataset). Code license TBD.