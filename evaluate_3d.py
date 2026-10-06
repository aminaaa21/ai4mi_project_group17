import nibabel as nib
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, distance_transform_edt
from pathlib import Path
from utils import class2one_hot, dice_batch # 3D dice already implemented in utils

"""
To evaluate 3d dataset, run the following in git bash

python evaluate_3d.py \
    --pred_dir TODO \
    --gt_dir TODO \
    --output_dir TODO

Each patient has gt and pred of which each is a 2D array 
-> this evaluator will load both and get the actual voxel labels
=> need to turn these integer labels into one-hot format (why? because dice_batch expects one-hot tensors)
"""
K = 5

CLASS_NAMES = ["background",
               "esophagus",
               "heart",
               "trachea",
               "aorta",
               ]

def load_nifti(path):
    nii = nib.load(str(path))

    image = np.asarray(nii.dataobj).astype(np.int64)
    spacing = nii.header.get_zooms()[:3]

    return image, spacing




def dice_3d(gt, pred):
    """
    3D Dice for all classes using dice_batch() from utils.py
    """

    gt = torch.from_numpy(gt).long().unsqueeze(0)
    pred = torch.from_numpy(pred).long().unsqueeze(0)

    gt = class2one_hot(gt, K)
    pred = class2one_hot(pred, K)

    return dice_batch(gt, pred).numpy()


def get_surface(mask):
    """Surface voxels of a 3D binary mask."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(mask)

    return mask & ~eroded


def get_crop(mask1, mask2):
    """
    Get a bounding box containing both masks.

    A small 1-voxel margin is added so that surface voxels
    near the boundary are handled correctly.
    """

    combined = mask1 | mask2

    coords = np.where(combined)

    if len(coords[0]) == 0:
        return None

    slices = []

    for axis in range(3):
        start = max(coords[axis].min() - 1, 0)
        end = min(coords[axis].max() + 2, combined.shape[axis])

        slices.append(slice(start, end))

    return tuple(slices)


def surface_distances(gt, pred, spacing):
    """
    Calculate symmetric surface distances once.

    These distances are then used for both HD95 and MSD.
    """

    if not np.any(gt) and not np.any(pred):
        return np.array([])

    if not np.any(gt) or not np.any(pred):
        return None

    gt_surface = get_surface(gt)
    pred_surface = get_surface(pred)

    crop = get_crop(
        gt_surface,
        pred_surface
    )

    gt_surface = gt_surface[crop]
    pred_surface = pred_surface[crop]

    gt_distance = distance_transform_edt(
        ~pred_surface,
        sampling=spacing
    )

    pred_distance = distance_transform_edt(
        ~gt_surface,
        sampling=spacing
    )

    distances = np.concatenate([
        gt_distance[gt_surface],
        pred_distance[pred_surface]
    ])

    return distances


def hd95_from_distances(distances):
    """Calculate HD95 from symmetric surface distances."""

    if distances is None:
        return np.inf

    if len(distances) == 0:
        return 0.0

    return np.percentile(
        distances,
        95
    )


def msd_from_distances(distances):
    """Calculate MSD from symmetric surface distances."""

    if distances is None:
        return np.inf

    if len(distances) == 0:
        return 0.0

    return np.mean(distances)

def find_gt(gt_dir, patient_id):
    """
    Find the GT file for a patient.

    Expected structure:

        gt_dir/
            Patient_01/
                GT.nii.gz
            Patient_02/
                GT.nii.gz
    """

    path = gt_dir / patient_id / "GT.nii.gz"

    if path.exists():
        return path

    path = gt_dir / patient_id / "GT.nii"

    if path.exists():
        return path

    return None

def make_plot(patient_ids, dice, hd95_values, msd_values, output_path):
    x = np.arange(len(patient_ids))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    metrics = [
        (dice, "Dice", "Higher is better"),
        (hd95_values, "HD95 (mm)", "Lower is better"),
        (msd_values, "MSD (mm)", "Lower is better"),
    ]

    for ax, (values, title, direction) in zip(axes, metrics):

        # Plot mean across the 5 classes for each patient
        patient_mean = np.mean(values, axis=1)
        overall_mean = np.mean(values)

        ax.bar(x, patient_mean)
        ax.axhline(
            overall_mean,
            linestyle="--",
            label=f"Overall mean = {overall_mean:.3f}",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(patient_ids, rotation=45)
        ax.set_title(f"{title} — {direction}")
        ax.set_ylabel(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main(args):

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions = sorted(
        pred_dir.glob("*.nii.gz")
    )

    if len(predictions) == 0:
        raise RuntimeError(
            f"No .nii.gz files found in {pred_dir}"
        )

    dice_results = {}
    hd95_results = {}
    msd_results = {}

    for i, pred_path in enumerate(predictions, 1):
        print(f"Evaluating {i}/{len(predictions)}", flush=True)

        patient_id = pred_path.stem.replace(
            ".nii",
            ""
        )
        print("got path", flush=True)

        gt_path = find_gt(
            gt_dir,
            patient_id
        )

        if gt_path is None:
            print(
                f"Skipping {patient_id}: "
                "GT not found."
            )
            continue

        print(f"\n{patient_id}")
        print("loaded nifti")
        pred, pred_spacing = load_nifti(
            pred_path
        )

        gt, gt_spacing = load_nifti(
            gt_path
        )
        print("check pred and gt same 3d shape")
        # Check that prediction and GT have
        # the same 3D shape.
        assert pred.shape == gt.shape, (
            f"{patient_id}: "
            f"{pred.shape} != {gt.shape}"
        )
        print("check voxel spacing")
        # Check voxel spacing.
        assert np.allclose(
            pred_spacing,
            gt_spacing
        ), (
            f"{patient_id}: prediction and GT "
            "have different spacing"
        )

        # SegTHOR labels are 0, 1, 2, 3, 4.
        assert set(np.unique(gt)).issubset(
            set(range(K))
        ), (
            f"{patient_id}: invalid GT labels "
            f"{np.unique(gt)}"
        )

        assert set(np.unique(pred)).issubset(
            set(range(K))
        ), (
            f"{patient_id}: invalid prediction labels "
            f"{np.unique(pred)}"
        )

        # ----------------------------------------
        # Dice
        # ----------------------------------------
        print("calculating dice")
        dice = dice_3d(
            gt,
            pred
        )

        # ----------------------------------------
        # HD95 and MSD
        # ----------------------------------------
        print("calculating hd95 and msd")

        hd95_values = np.zeros(K)
        msd_values = np.zeros(K)

        for class_id in range(K):

            print(
                f"  {CLASS_NAMES[class_id]}",
                flush=True
            )

            gt_class = gt == class_id
            pred_class = pred == class_id

            # Calculate the expensive distance transforms once.
            distances = surface_distances(
                gt_class,
                pred_class,
                gt_spacing
            )

            # Get both metrics from those same distances.
            hd95_values[class_id] = hd95_from_distances(
                distances
            )

            msd_values[class_id] = msd_from_distances(
                distances
            )
        print("\nDice:")

        for class_id in range(K):
            print(
                f"  {CLASS_NAMES[class_id]}: "
                f"{dice[class_id]:.4f}"
            )

        print("HD95:")

        for class_id in range(K):
            print(
                f"  {CLASS_NAMES[class_id]}: "
                f"{hd95_values[class_id]:.4f} mm"
            )

        print("MSD:")

        for class_id in range(K):
            print(
                f"  {CLASS_NAMES[class_id]}: "
                f"{msd_values[class_id]:.4f} mm"
            )
    if len(dice_results) == 0:
        raise RuntimeError(
            "No patients were evaluated."
        )

    # ----------------------------------------
    # Mean over patients
    # ----------------------------------------

    dice_array = np.array(
        list(dice_results.values())
    )

    hd95_array = np.array(
        list(hd95_results.values())
    )

    msd_array = np.array(
        list(msd_results.values())
    )

    print("\n==============================")
    print("MEAN OVER PATIENTS")
    print("==============================")

    print("\nDice:")
    for class_id in range(K):
        print(
            f"  {CLASS_NAMES[class_id]}: "
            f"{dice_array[:, class_id].mean():.4f}"
        )

    print("\nHD95:")
    for class_id in range(K):
        print(
            f"  {CLASS_NAMES[class_id]}: "
            f"{hd95_array[:, class_id].mean():.4f} mm"
        )

    print("\nMSD:")
    for class_id in range(K):
        print(
            f"  {CLASS_NAMES[class_id]}: "
            f"{msd_array[:, class_id].mean():.4f} mm"
        )

    # ----------------------------------------
    # Save results
    # ----------------------------------------

    np.savez(
        output_dir / "dice.npz",
        **dice_results
    )

    np.savez(
        output_dir / "hd95.npz",
        **hd95_results
    )

    np.savez(
        output_dir / "msd.npz",
        **msd_results
    )

    print(
        f"\nSaved metrics to {output_dir}"
    )
    # ----------------------------------------
    # Make plot
    # ----------------------------------------

    make_plot(
        list(dice_results.keys()),
        dice_array,
        hd95_array,
        msd_array,
        output_dir / "metrics.png"
    )

    print(
        f"\nSaved metrics to {output_dir}"
    )

    print(
        f"Saved plot to {output_dir / 'metrics.png'}"
    )

def get_args():

    parser = argparse.ArgumentParser(
        description="Evaluate 3D SegTHOR predictions"
    )

    parser.add_argument(
        "--pred_dir",
        type=Path,
        required=True,
        help="Folder containing stitched 3D predictions"
    )

    parser.add_argument(
        "--gt_dir",
        type=Path,
        required=True,
        help="Folder containing patient GT folders"
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/metrics"),
        help="Folder to save metrics"
    )

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())