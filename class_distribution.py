from pathlib import Path

import numpy as np
from PIL import Image


gt_dir = Path("data/SEGTHOR/train/gt")

class_names = [
    "background",
    "esophagus+aorta",
    "heart",
    "trachea"
]

class_counts = np.zeros(4, dtype=np.int64)

crop_size = 192


# Check which pixel values are present in the masks
unique_values = set()

for gt_path in sorted(gt_dir.glob("*.png")):
    mask = np.array(Image.open(gt_path))
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

    # Calculate where the center crop starts
    height, width = mask.shape

    start_y = (height - crop_size) // 2
    start_x = (width - crop_size) // 2

    end_y = start_y + crop_size
    end_x = start_x + crop_size

    # Apply the same center crop used during training
    mask = mask[start_y:end_y, start_x:end_x]

    # Convert the mask values to class numbers
    labels = (mask / 63).astype(int)

    # Count how many pixels belong to every class
    for class_id in range(4):
        class_counts[class_id] += np.sum(labels == class_id)


# Calculate the total number of pixels
total_pixels = class_counts.sum()


# Print the class distribution
print("Class distribution after training preprocessing:")

for class_id in range(4):
    percentage = 100 * class_counts[class_id] / total_pixels

    print(
        class_id,
        class_names[class_id],
        class_counts[class_id],
        f"{percentage:.4f}%"
    )


# Calculate the frequency of every class
class_frequencies = class_counts / total_pixels

# Give rare classes more weight
# Square root keeps the difference between weights less extreme
class_weights = 1 / np.sqrt(class_frequencies)

# Normalize so the average weight is 1
class_weights = class_weights / class_weights.mean()


print()
print("Class weights:")

for class_id in range(4):
    print(
        class_id,
        class_names[class_id],
        f"{class_weights[class_id]:.4f}"
    )