#!/usr/bin/env python3

# Compare several methods (baseline, HU windows, ...), separately for contrast-enhanced and
# non-enhanced patients.
#
# The 2D .png slices (predictions and ground truth) are stacked back into one 3D volume per
# patient, and we compute the 3D Dice for every class. Several folders can be given per method,
# which is handy for cross-validation: every patient is in the validation set of exactly one fold.
# The first method is the reference: the summary shows the difference of the others with it.

import re
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ["background", "esophagus", "heart", "trachea", "aorta"]
K = len(CLASS_NAMES)


def load_volumes(folders: list[Path]) -> dict[str, np.ndarray]:
    # Group the .png slices by patient (file names look like Patient_01_0042.png)
    slices: dict[str, list[Path]] = {}
    for folder in folders:
        pngs: list[Path] = sorted(folder.glob("*.png"))
        assert len(pngs) > 0, f"No .png found in {folder}"
        for png in pngs:
            patient: str = re.match(r"(Patient_\d\d)_\d{4}", png.stem).group(1)
            if patient not in slices:
                slices[patient] = []
            slices[patient].append(png)

    volumes: dict[str, np.ndarray] = {}
    for patient, paths in slices.items():
        paths = sorted(paths)  # Sorted by slice number
        volume = np.stack([np.array(Image.open(p)) for p in paths])
        # The classes are saved as {0, 63, 126, 189, 252}, map them back to {0, 1, 2, 3, 4}
        volumes[patient] = volume // 63
        assert volumes[patient].max() < K

    return volumes


def dice_3d(gt: np.ndarray, pred: np.ndarray, k: int) -> float:
    gt_k = gt == k
    pred_k = pred == k

    # The Dice is not defined if the organ is absent from the ground truth
    if gt_k.sum() == 0:
        return float('nan')

    intersection = (gt_k & pred_k).sum()
    return 2 * intersection / (gt_k.sum() + pred_k.sum())


def read_contrast_csv(path: Path) -> dict[str, int]:
    contrast: dict[str, int] = {}
    with open(path) as f:
        next(f)  # Skip the header
        for line in f:
            patient, _, is_contrast = line.strip().split(",")
            contrast[patient] = int(is_contrast)
    return contrast


def main(args: argparse.Namespace) -> None:
    contrast: dict[str, int] = read_contrast_csv(args.contrast_csv)

    gts: dict[str, np.ndarray] = load_volumes(args.gt_dirs)

    # args.method looks like [["baseline", dir1, dir2, ...], ["window", dir1, dir2, ...], ...]
    preds: dict[str, dict[str, np.ndarray]] = {}
    for method_args in args.method:
        name: str = method_args[0]
        folders: list[Path] = [Path(f) for f in method_args[1:]]
        preds[name] = load_volumes(folders)
    methods: list[str] = list(preds.keys())
    reference: str = methods[0]

    patients: list[str] = sorted(gts.keys())
    print(f"Found {len(patients)} patients in the ground truth")

    # dices[method][patient] = array of K Dice values (one per class)
    dices: dict[str, dict[str, np.ndarray]] = {}
    for method, method_preds in preds.items():
        assert sorted(method_preds.keys()) == patients, (method, sorted(method_preds.keys()))
        dices[method] = {}
        for patient in patients:
            gt = gts[patient]
            pred = method_preds[patient]
            assert gt.shape == pred.shape, (method, patient, gt.shape, pred.shape)
            dices[method][patient] = np.array([dice_3d(gt, pred, k) for k in range(K)])

    # Only report the foreground classes that are present in at least one ground truth
    classes: list[int] = [k for k in range(1, K)
                          if not np.isnan([dices[reference][p][k] for p in patients]).all()]
    names: list[str] = [CLASS_NAMES[k] for k in classes]

    # Per-patient results
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.dest, 'w') as f:
        f.write("patient,contrast,method," + ",".join(names) + ",mean\n")
        for patient in patients:
            for method in dices:
                values = dices[method][patient][classes]
                f.write(f"{patient},{contrast[patient]},{method},"
                        + ",".join(f"{v:.4f}" for v in values) + f",{np.nanmean(values):.4f}\n")
    print(f"Saved per-patient 3D Dice to {args.dest}")

    # Summary: average over the patients of each group
    groups: dict[str, list[str]] = {"contrast-enhanced": [p for p in patients if contrast[p] == 1],
                                    "non-enhanced": [p for p in patients if contrast[p] == 0],
                                    "all": patients}

    lines: list[str] = []
    header: str = f"{'group':<18} {'n':>3}  {'method':<13}" + "".join(f"{n:>11}" for n in names) + f"{'mean':>11}"
    lines.append(f"3D Dice, averaged over the patients of each group (diff = method - {reference})")
    lines.append(header)
    lines.append("-" * len(header))
    for group, group_patients in groups.items():
        if len(group_patients) == 0:
            lines.append(f"{group:<18} {0:>3}  (no patient)")
            continue

        means: dict[str, np.ndarray] = {}
        for method in methods:
            per_patient = np.array([dices[method][p][classes] for p in group_patients])
            per_class = np.nanmean(per_patient, axis=0)
            means[method] = np.append(per_class, per_class.mean())

        rows: list[tuple[str, np.ndarray, str]] = [(method, means[method], "") for method in methods]
        for method in methods[1:]:
            rows.append((f"diff {method}", means[method] - means[reference], "+"))

        for i, (row_name, values, sign) in enumerate(rows):
            group_str = f"{group:<18} {len(group_patients):>3}" if i == 0 else " " * 22
            lines.append(f"{group_str}  {row_name:<13}" + "".join(f"{v:>{sign}11.3f}" for v in values))
        lines.append("")

    summary: str = "\n".join(lines)
    print(summary)
    summary_path: Path = args.dest.with_suffix(".txt")
    with open(summary_path, 'w') as f:
        f.write(summary + "\n")
    print(f"Saved summary to {summary_path}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D Dice for contrast-enhanced vs non-enhanced patients")
    parser.add_argument('--contrast_csv', type=Path, required=True,
                        help="Output of detect_contrast.py")
    parser.add_argument('--gt_dirs', type=Path, nargs='+', required=True,
                        help="Folders with the ground truth .png, e.g. data/SEGTHOR_WINDOW_fold0/val/gt")
    parser.add_argument('--method', nargs='+', action='append', required=True,
                        metavar=('NAME', 'DIR'),
                        help="A method name followed by its prediction folders, e.g. "
                             "--method baseline results/.../baseline_fold0/best_epoch/val ... "
                             "Repeat --method for every method; the first one is the reference.")
    parser.add_argument('--dest', type=Path, required=True,
                        help="CSV file for the per-patient Dice (a .txt summary is saved next to it)")
    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
