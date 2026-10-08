from pathlib import Path

import numpy as np
from PIL import Image


gt_dir = Path("data/SEGTHOR_FULL/train/gt")

if not gt_dir.exists():
    raise FileNotFoundError(f"Could not find training masks at: {gt_dir}")

class_names = [
    "background",
    "esophagus",
    "heart",
    "trachea",
    "aorta"
]

class_counts = np.zeros(5, dtype=np.int64)


# Check which pixel values are present in the masks
unique_values = set()

for gt_path in sorted(gt_dir.glob("*.png")):
    mask = np.array(Image.open(gt_path))

    # Skip empty training slices
    if mask.sum() == 0:
        continue

    unique_values.update(np.unique(mask).tolist())

print("Unique values in training masks:")
print(sorted(unique_values))
print()


# Go through all training masks
for gt_path in sorted(gt_dir.glob("*.png")):
    mask = np.array(Image.open(gt_path))

    # Skip empty training slices
    if mask.sum() == 0:
        continue

    # No crop here:
    # Galina's clean baseline used --crop_size 0

    # Convert mask values to class numbers
    labels = (mask / 63).astype(int)

    # Count pixels for all five classes
    for class_id in range(5):
        class_counts[class_id] += np.sum(labels == class_id)


# Calculate total number of pixels
total_pixels = class_counts.sum()


# Print class distribution
print("Class distribution after empty-slice filtering:")

for class_id in range(5):
    percentage = 100 * class_counts[class_id] / total_pixels

    print(
        class_id,
        class_names[class_id],
        class_counts[class_id],
        f"{percentage:.4f}%"
    )


# Calculate frequency of every class
class_frequencies = class_counts / total_pixels

# Give rare classes more weight
# Square root keeps the difference between weights less extreme
class_weights = 1 / np.sqrt(class_frequencies)

# Normalize so the average weight is 1
class_weights = class_weights / class_weights.mean()


print()
print("Class weights:")

for class_id in range(5):
    print(
        class_id,
        class_names[class_id],
        f"{class_weights[class_id]:.4f}"
    )