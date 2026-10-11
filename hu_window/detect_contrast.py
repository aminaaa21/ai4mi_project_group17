#!/usr/bin/env python3

# Find out which patients have a contrast-enhanced CT scan.
#
# The contrast agent (iodine) is injected in the blood, which makes the blood much brighter on CT:
#   - blood without contrast: around 30-50 HU
#   - blood with contrast:    well above 100 HU
# The heart (label 2) is mostly filled with blood, so we look at the median HU inside the
# heart label, and call the scan "contrast-enhanced" when it is above a threshold.

import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import nibabel as nib

HEART_LABEL = 2


def heart_median_hu(patient_dir: Path) -> float:
    ct = np.asarray(nib.load(str(patient_dir / f"{patient_dir.name}.nii.gz")).dataobj)
    gt = np.asarray(nib.load(str(patient_dir / "GT.nii.gz")).dataobj)
    assert ct.shape == gt.shape

    heart_voxels = ct[gt == HEART_LABEL]
    assert len(heart_voxels) > 0, f"No heart label for {patient_dir.name}"

    return float(np.median(heart_voxels))


def main(args: argparse.Namespace) -> None:
    patient_dirs: list[Path] = sorted(p for p in (args.source_dir / "train").iterdir() if p.is_dir())
    print(f"Found {len(patient_dirs)} patients")

    with Pool(args.process) as pool:
        medians: list[float] = pool.map(heart_median_hu, patient_dirs)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.dest, 'w') as f:
        f.write("patient,heart_median_hu,contrast\n")
        for patient_dir, median in zip(patient_dirs, medians):
            contrast: int = 1 if median > args.threshold else 0
            f.write(f"{patient_dir.name},{median:.1f},{contrast}\n")
            print(f"{patient_dir.name}: heart median {median:6.1f} HU -> "
                  f"{'contrast-enhanced' if contrast else 'non-enhanced'}")

    n_contrast: int = sum(1 for m in medians if m > args.threshold)
    print(f"{n_contrast} contrast-enhanced, {len(medians) - n_contrast} non-enhanced")
    print(f"Saved to {args.dest}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect contrast-enhanced CT scans")
    parser.add_argument('--source_dir', type=Path, required=True,
                        help="Folder with the nifti files, e.g. data/segthor_part1")
    parser.add_argument('--dest', type=Path, required=True,
                        help="CSV file to write, e.g. results/hu_window/contrast.csv")
    parser.add_argument('--threshold', type=float, default=80,
                        help="Median heart HU above which a scan counts as contrast-enhanced")
    parser.add_argument('--process', '-p', type=int, default=4,
                        help="The number of cores to use for processing")
    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
