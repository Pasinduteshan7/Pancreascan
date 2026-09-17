# src/plot_training.py
"""
Plot training curves for both stages of the PancreaScan pipeline.

Reads the loss history JSON files saved during training and produces
publication-ready training curve plots.

Usage:
    python plot_training.py
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HISTORY_DIR = "../outputs"
OUTPUT_PATH = "../outputs/training_curves.png"


def load_history(stage_name):
    """Load training history for a stage, or return None if unavailable."""
    path = os.path.join(HISTORY_DIR, f"training_history_{stage_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_training_curves():
    s1 = load_history("stage1")
    s2 = load_history("stage2")

    if s1 is None and s2 is None:
        print("No training history files found.")
        print("Run train_stage1.py and/or train_stage2.py to generate them.")
        print(f"Expected files: {HISTORY_DIR}/training_history_stage1.json")
        return

    # Dark theme matching the project's visual style
    plt.style.use('dark_background')

    num_plots = sum(1 for h in [s1, s2] if h is not None)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))
    fig.patch.set_facecolor('#060a14')

    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    for stage_name, history, color_train, color_val in [
        ("Stage 1 — Pancreas Localization", s1, '#33d4ab', '#ffb347'),
        ("Stage 2 — Tumor Detection", s2, '#5ea4f5', '#ff4757'),
    ]:
        if history is None:
            continue

        ax = axes[plot_idx]
        ax.set_facecolor('#0c1221')

        epochs = range(1, len(history['train_loss']) + 1)

        ax.plot(epochs, history['train_loss'], '-o', color=color_train,
                linewidth=2, markersize=4, label='Train Loss', alpha=0.9)
        ax.plot(epochs, history['val_loss'], '-s', color=color_val,
                linewidth=2, markersize=4, label='Val Loss', alpha=0.9)

        # Mark the best validation epoch
        best_epoch = int(np.argmin(history['val_loss'])) + 1
        best_val = min(history['val_loss'])
        ax.axvline(x=best_epoch, color='#666', linestyle='--', alpha=0.5)
        ax.annotate(f'Best: {best_val:.4f}\n(epoch {best_epoch})',
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + 0.5, best_val + 0.02),
                    color='#aaa', fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='#666'))

        ax.set_xlabel('Epoch', color='#aaa', fontsize=11)
        ax.set_ylabel('Loss', color='#aaa', fontsize=11)
        ax.set_title(stage_name, color='#e8ecf4', fontsize=13, pad=12)
        ax.legend(framealpha=0.3, fontsize=10)
        ax.tick_params(colors='#888')
        ax.grid(True, alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333')
        ax.spines['bottom'].set_color('#333')

        plot_idx += 1

    fig.suptitle('PancreaScan Training Curves', color='#e8ecf4',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(pad=2)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Training curves saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_training_curves()
