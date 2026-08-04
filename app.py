# app.py — PancreaScan Demo Web Server (Two-Stage Pipeline)
import os
import io
import base64
import json
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import from src/
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model import UNet
from pipeline import run_stage1, run_stage2, combine_predictions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STAGE1_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'outputs', 'best_model_stage1.pth')
STAGE2_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'outputs', 'best_model_stage2.pth')
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'images')
LABELS_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'labels')
SPLIT_FILE  = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'split.json')
WEB_DIR     = os.path.join(os.path.dirname(__file__), 'web')

STAGE1_INPUT_SIZE = 256
STAGE2_INPUT_SIZE = 128
CROP_MARGIN = 20

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=WEB_DIR)
CORS(app)

# ---------------------------------------------------------------------------
# Load both models once at startup
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[PancreaScan] Loading models on {device}...")

model_s1 = UNet(in_channels=1, num_classes=2).to(device)
model_s1.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=device, weights_only=True))
model_s1.eval()
print("[PancreaScan] Stage 1 model loaded.")

model_s2 = UNet(in_channels=1, num_classes=3).to(device)
model_s2.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location=device, weights_only=True))
model_s2.eval()
print("[PancreaScan] Stage 2 model loaded. Ready.")

# ---------------------------------------------------------------------------
# Image helpers → base64 PNG
# ---------------------------------------------------------------------------

def array_to_b64(arr, cmap='gray', vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def stage1_mask_to_b64(stage1_pred_256, bbox=None):
    """Render Stage 1 binary mask with bounding box."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.set_facecolor('#060a14')
    bg = np.zeros((STAGE1_INPUT_SIZE, STAGE1_INPUT_SIZE))
    ax.imshow(bg, cmap='gray', vmin=0, vmax=1)

    overlay = np.zeros((*stage1_pred_256.shape, 4), dtype=np.float32)
    overlay[stage1_pred_256 == 1] = [0.2, 0.83, 0.67, 0.7]  # teal
    ax.imshow(overlay)

    if bbox is not None:
        h, w = stage1_pred_256.shape
        # bbox is in original coords — rescale to 256x256 display
        # We'll just show bbox passed in 256-space (already computed in run_stage1)
        rmin, rmax, cmin, cmax = bbox
        from matplotlib.patches import Rectangle
        rect = Rectangle((cmin, rmin), cmax - cmin, rmax - rmin,
                          linewidth=2, edgecolor='#ffb347', facecolor='none')
        ax.add_patch(rect)

    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor('#060a14')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def crop_to_b64(crop):
    """Render the zoomed pancreas crop."""
    return array_to_b64(crop, cmap='gray')


def overlay_to_b64(ct_slice, mask):
    """CT slice with teal=pancreas, red=tumor overlay."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(ct_slice, cmap='gray')
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask == 1] = [0.2, 0.83, 0.67, 0.55]
    rgba[mask == 2] = [1.0, 0.28, 0.34, 0.65]
    ax.imshow(rgba)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def confidence_to_b64(stage2_probs_crop):
    """Inferno heatmap of tumor probability (class 2)."""
    tumor_prob = stage2_probs_crop[2]  # (128, 128)
    return array_to_b64(tumor_prob, cmap='inferno', vmin=0, vmax=1)


# ---------------------------------------------------------------------------
# Two-stage inference
# ---------------------------------------------------------------------------

def run_pipeline_inference(image_np):
    """
    Full two-stage pipeline on a raw CT slice (numpy, 0-1 windowed).
    Returns all images and stats needed for the frontend.
    """
    original_h, original_w = image_np.shape

    # Stage 1 — pancreas localization
    stage1_pred_256, bbox = run_stage1(image_np, model_s1, device)

    if bbox is None:
        # Resize image for display
        t = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0)
        img256 = F.interpolate(t, size=(256, 256), mode='bilinear', align_corners=False)[0,0].numpy()
        return {
            'ct_image':      array_to_b64(img256, 'gray'),
            'stage1_image':  stage1_mask_to_b64(stage1_pred_256, bbox=None),
            'crop_image':    None,
            'overlay_image': array_to_b64(img256, 'gray'),
            'confidence_image': None,
            'ground_truth_image': None,
            'has_ground_truth': False,
            'stats': {
                'pancreas_detected': False,
                'tumor_detected': False,
                'max_tumor_confidence': 0.0,
                'background_pct': 100.0,
                'pancreas_pct': 0.0,
                'tumor_pct': 0.0,
            }
        }

    # Stage 2 — fine-grained detection on crop
    stage2_pred_crop, stage2_probs_crop, crop = run_stage2(image_np, bbox, model_s2, device)

    # Combine back to full slice
    full_pred = combine_predictions(image_np, stage1_pred_256, stage2_pred_crop, bbox)

    # Resize original image to 256x256 for display
    t = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0)
    img256 = F.interpolate(t, size=(256, 256), mode='bilinear', align_corners=False)[0,0].numpy()

    # Scale full_pred to 256x256 for overlay display
    pred_t = torch.from_numpy(full_pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pred256 = F.interpolate(pred_t, size=(256, 256), mode='nearest')[0,0].numpy().astype(int)

    # Compute bbox in 256-space for stage1 display
    rmin, rmax, cmin, cmax = bbox
    scale_r = STAGE1_INPUT_SIZE / original_h
    scale_c = STAGE1_INPUT_SIZE / original_w
    bbox_256 = (int(rmin * scale_r), int(rmax * scale_r),
                int(cmin * scale_c), int(cmax * scale_c))

    total = full_pred.size
    stats = {
        'pancreas_detected':    bool(np.any(full_pred == 1)),
        'tumor_detected':       bool(np.any(full_pred == 2)),
        'max_tumor_confidence': float(np.max(stage2_probs_crop[2])),
        'background_pct':       float(np.sum(full_pred == 0) / total * 100),
        'pancreas_pct':         float(np.sum(full_pred == 1) / total * 100),
        'tumor_pct':            float(np.sum(full_pred == 2) / total * 100),
    }

    return {
        'ct_image':           array_to_b64(img256, 'gray'),
        'stage1_image':       stage1_mask_to_b64(stage1_pred_256, bbox_256),
        'crop_image':         crop_to_b64(crop),
        'overlay_image':      overlay_to_b64(img256, pred256),
        'confidence_image':   confidence_to_b64(stage2_probs_crop),
        'ground_truth_image': None,
        'has_ground_truth':   False,
        'stats':              stats,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def serve_index():
    return send_from_directory(WEB_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(WEB_DIR, path)


@app.route('/api/samples', methods=['GET'])
def get_samples():
    try:
        with open(SPLIT_FILE) as f:
            splits = json.load(f)
        val_files = splits.get('val', [])
        samples = []
        for filename in val_files:
            label_path = os.path.join(LABELS_DIR, filename)
            if os.path.exists(label_path):
                label = np.load(label_path)
                has_pancreas = bool(np.any(label == 1))
                has_tumor    = bool(np.any(label == 2))
                if has_pancreas or has_tumor:
                    samples.append({
                        'filename':    filename,
                        'has_pancreas': has_pancreas,
                        'has_tumor':    has_tumor,
                    })
            if len(samples) >= 20:
                break
        return jsonify({'samples': samples})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename.endswith('.npy'):
        return jsonify({'error': 'Only .npy files are supported'}), 400
    try:
        image_np = np.load(io.BytesIO(file.read()))
        result = run_pipeline_inference(image_np)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-sample', methods=['POST'])
def predict_sample():
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400

    filename   = data['filename']
    image_path = os.path.join(IMAGES_DIR, filename)
    label_path = os.path.join(LABELS_DIR, filename)

    if not os.path.exists(image_path):
        return jsonify({'error': f'Sample not found: {filename}'}), 404

    try:
        image_np = np.load(image_path)
        result   = run_pipeline_inference(image_np)

        # Add ground truth overlay if label exists
        if os.path.exists(label_path):
            label_np = np.load(label_path)
            t = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0)
            img256 = F.interpolate(t, size=(256, 256), mode='bilinear', align_corners=False)[0,0].numpy()
            lbl_t  = torch.from_numpy(label_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            lbl256 = F.interpolate(lbl_t, size=(256, 256), mode='nearest')[0,0].numpy().astype(int)
            result['ground_truth_image'] = overlay_to_b64(img256, lbl256)
            result['has_ground_truth']   = True

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("[PancreaScan] Starting web server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
