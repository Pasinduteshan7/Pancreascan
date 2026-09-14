# src/train_stage2.py
import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_stage2 import Stage2Dataset
from model import UNet

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "outputs", "checkpoint_stage2.pth")
BEST_MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "best_model_stage2.pth")


def train(split_file=None, fresh=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if split_file is None:
        split_file = os.path.join(PROJECT_DIR, "data", "processed", "split.json")
    print(f"Using split file: {split_file}")

    train_ds = Stage2Dataset(split='train', split_file=split_file)
    val_ds = Stage2Dataset(split='val', split_file=split_file)

    print(f"Stage 2 — Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Larger batch size — 128x128 crops are small, so we can fit more
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    # 3 classes: background (within crop), healthy pancreas, tumor
    model = UNet(in_channels=1, num_classes=3).to(device)

    # Cancer is rare even within the cropped pancreas region — weight it more heavily
    class_weights = torch.tensor([0.1, 1.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    start_epoch = 0
    best_val_loss = float('inf')
    os.makedirs(os.path.join(PROJECT_DIR, "outputs"), exist_ok=True)

    # Loss history for training curves
    history_path = os.path.join(PROJECT_DIR, "outputs", "training_history_stage2.json")
    if os.path.exists(history_path) and not fresh:
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = {'train_loss': [], 'val_loss': []}

    # Resume from checkpoint if it exists (unless --fresh)
    if os.path.exists(CHECKPOINT_PATH) and not fresh:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}")
    elif fresh:
        print("Fresh training (ignoring any existing checkpoints)")

    num_epochs = 20

    for epoch in range(start_epoch, num_epochs):
        # --- Training phase ---
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # Log losses for training curves
        history['train_loss'].append(round(avg_train_loss, 6))
        history['val_loss'].append(round(avg_val_loss, 6))
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Save checkpoint every epoch (resume-safe)
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, CHECKPOINT_PATH)

        # Save best model separately
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> Saved new best Stage 2 model (val_loss={avg_val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 2 (tumor detection)")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Path to split JSON file (default: split.json, use split_combined.json for combined training)")
    parser.add_argument("--fresh", action="store_true",
                        help="Train from scratch, ignoring existing checkpoints")
    args = parser.parse_args()
    train(split_file=args.split_file, fresh=args.fresh)
