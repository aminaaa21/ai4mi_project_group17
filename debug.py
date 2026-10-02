

import numpy as np
from pathlib import Path
from PIL import Image

dest_dir = Path("./results/2d_preprocessed_cpu")

# 1. Check loss and dice logs
if (dest_dir / "dice_val.npy").exists():
    dice_val = np.load(dest_dir / "dice_val.npy")
    loss_val = np.load(dest_dir / "loss_val.npy")
    print(f"Dice val shape: {dice_val.shape} (Epochs, Samples, Classes)")
    print(f"Max validation Dice across all epochs: {dice_val[:, :, 1:].mean():.4f}")
else:
    print("Validation numpy logs not found yet.")

# 2. Check class distribution of the latest saved prediction images
pred_dir = dest_dir / "iter019" / "val" # or check best_epoch/val if it exists
if pred_dir.exists():
    pred_files = list(pred_dir.glob("*.png"))
    if pred_files:
        sample_pred = Image.open(pred_files[0])
        arr = np.array(sample_pred)
        unique, counts = np.unique(arr, return_counts=True)
        print("\nPrediction pixel value distribution for a validation sample:")
        for val, count in zip(unique, counts):
            print(f"  Pixel value {val}: {count} pixels")
    else:
        print("No prediction images found in iter019/val.")
else:
    print(f"Prediction directory {pred_dir} not found.")