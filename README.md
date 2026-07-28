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

## Status
🚧 In progress — data pipeline complete, model training next.

## License
CC-BY-SA 4.0 (dataset). Code license TBD.