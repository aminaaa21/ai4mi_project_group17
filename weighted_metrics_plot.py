import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Number of completed epochs is passed by main.py
completed_epochs = int(sys.argv[1])

# Folder of the run (also passed by main.py), so the plots end up next to the run they belong to
if len(sys.argv) > 2:
    results_dir = Path(sys.argv[2])
else:
    results_dir = Path("results/weighted_dice_lr5e5_wd1e5_50ep")

loss_tra = np.load(results_dir / "loss_tra.npy")[:completed_epochs]
loss_val = np.load(results_dir / "loss_val.npy")[:completed_epochs]
dice_tra = np.load(results_dir / "dice_tra.npy")[:completed_epochs]
dice_val = np.load(results_dir / "dice_val.npy")[:completed_epochs]
acc_val = np.load(results_dir / "acc_val.npy")[:completed_epochs]

epochs = np.arange(1, completed_epochs + 1)

# Mean values per epoch
mean_loss_tra = loss_tra.mean(axis=1)
mean_loss_val = loss_val.mean(axis=1)

class_dice_tra = dice_tra.mean(axis=1)
class_dice_val = dice_val.mean(axis=1)

mean_dice_tra = class_dice_tra[:, 1:].mean(axis=1)
mean_dice_val = class_dice_val[:, 1:].mean(axis=1)

mean_acc_val = acc_val.mean(axis=1)


# 1. Train + validation loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, mean_loss_tra, label="Train Loss")
plt.plot(epochs, mean_loss_val, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Train and Validation Loss - Epochs 1-{completed_epochs}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    results_dir / f"loss_until_epoch_{completed_epochs:03d}.png",
    dpi=300
)
plt.close()


# 2. Validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(epochs, mean_acc_val, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Validation Accuracy - Epochs 1-{completed_epochs}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    results_dir / f"accuracy_until_epoch_{completed_epochs:03d}.png",
    dpi=300
)
plt.close()


# 3. Train vs validation mean Dice
plt.figure(figsize=(8, 5))
plt.plot(epochs, mean_dice_tra, label="Train Mean Dice")
plt.plot(epochs, mean_dice_val, label="Validation Mean Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title(f"Train vs Validation Dice - Epochs 1-{completed_epochs}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    results_dir / f"dice_until_epoch_{completed_epochs:03d}.png",
    dpi=300
)
plt.close()


# 4. Validation Dice per foreground class
class_names = {
    1: "Esophagus",
    2: "Heart",
    3: "Trachea",
    4: "Aorta",
}

plt.figure(figsize=(9, 6))

for c in range(1, class_dice_val.shape[1]):
    plt.plot(
        epochs,
        class_dice_val[:, c],
        label=class_names.get(c, f"Class {c}")
    )

plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title(f"Validation Dice per Class - Epochs 1-{completed_epochs}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    results_dir / f"validation_dice_classes_until_epoch_{completed_epochs:03d}.png",
    dpi=300
)
plt.close()

print(f"Saved plots through epoch {completed_epochs}")
