from pathlib import Path
import numpy as np
import nibabel as nib
from skimage.io import imsave
from skimage.transform import resize
import warnings

source_dir = Path("data/test")
dest_dir = Path("data/SEGTHOR/test/img")
dest_dir.mkdir(parents=True, exist_ok=True)

scans = sorted(source_dir.glob("*.nii.gz"))
if not scans:
    raise FileNotFoundError(f"No .nii.gz scans found in {source_dir.resolve()}")

for scan_path in scans:
    patient_id = scan_path.name.removesuffix(".nii.gz")
    nii = nib.load(str(scan_path))
    volume = np.asarray(nii.dataobj)

    # Match the normalization and resizing used by slice_segthor.py
    volume = volume.astype(np.float32)
    volume -= volume.min()
    max_val = volume.max()
    if max_val == 0:
        normalized = np.zeros_like(volume, dtype=np.uint8)
    else:
        normalized = (255 * volume / max_val).astype(np.uint8)

    for z in range(volume.shape[2]):
        slice_2d = resize(
            normalized[:, :, z],
            (256, 256),
            mode="constant",
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint8)

        out_path = dest_dir / f"{patient_id}_{z:04d}.png"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            imsave(str(out_path), slice_2d)

    print(f"Sliced {patient_id}: {volume.shape[2]} slices")

print(f"Done. Test PNGs saved to {dest_dir.resolve()}")
