# src/preprocessing.py
import numpy as np

def apply_window(slice_2d, center=40, width=400):
    """
    Clip CT slice to a soft-tissue HU window and normalize to 0-1.
    
    center=40, width=400 is a standard 'soft tissue' window,
    covering roughly -160 to 240 HU.
    """
    low = center - width // 2
    high = center + width // 2
    windowed = np.clip(slice_2d, low, high)
    normalized = (windowed - low) / (high - low)
    return normalized