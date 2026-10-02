import numpy as np
from dataset import SliceDataset
from torch.utils.data import DataLoader

# Load dataset with augmentations active, crop_size=None to get full augmented dimensions
dataset = SliceDataset(dest="results/2d_preprocessed", mode="train", augment=True, crop_size=None)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

samples_tested = 30
print("Testing crop sizes downward from 256 with augmentations...")

for crop_size in range(256, 160, -8):
    violations = 0
    for i, batch in enumerate(loader):
        if i >= samples_tested: 
            break
        gt = batch['gt'].numpy()
        
        H, W = gt.shape[-2:]
        start_h = (H - crop_size) // 2
        start_w = (W - crop_size) // 2
        
        cropped_gt = gt[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
        
        if gt.sum() != cropped_gt.sum():
            violations += 1
            
    print(f"Crop Size {crop_size}: Violations = {violations}/{samples_tested}")
    
    if violations > 0:
        print(f"--> Boundary reached! Minimum safe crop size is {crop_size + 8}.")
        break