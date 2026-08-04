# src/train_stage2.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_stage2 import Stage2Dataset
from model import UNet

CHECKPOINT_PATH = "../outputs/checkpoint_stage2.pth"
BEST_MODEL_PATH = "../outputs/best_model_stage2.pth"


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_ds = Stage2Dataset(split='train')
    val_ds = Stage2Dataset(split='val')

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
    os.makedirs("../outputs", exist_ok=True)

    # Resume from checkpoint if it exists
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}")

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
    train()
