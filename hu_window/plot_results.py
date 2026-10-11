#!/usr/bin/env python3

# Charts for the HU window experiment.
#
#   1_input_examples.png    the same slice with every normalization method
#   2_dice_change.png       per-patient change of 3D Dice compared to the baseline, per class and group
#   3_dice_per_patient.png  mean 3D Dice of every patient, for every method
#   4_contrast_gap.png      how much better contrast-enhanced patients do than non-enhanced ones
#   5_training_curves.png   validation Dice during training, for every fold
#
# The methods and classes are read from dice_3d.csv (written by evaluate_contrast.py).
# The first method in that file is the reference (the baseline).

import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # No window, only save the figures
import matplotlib.pyplot as plt
from PIL import Image

# Name shown in the charts and color of every method. The baseline is the reference, in gray;
# the other methods use colors of a colorblind-safe palette.
METHODS = {"baseline": ("Baseline (min-max)", "#8f8e89"),
           "window": ("Window -150..250 HU", "#2a78d6"),
           "wide": ("Window -150..400 HU", "#eb6834"),
           "multi": ("3 windows (3 channels)", "#1baf7a"),
           "multi_dice": ("3 windows + Dice loss", "#4a3aa7")}

# Methods trained on the sliced data of another method (only the loss is different)
DATA_OF = {"multi_dice": "multi"}

TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
LIGHT_GRAY = "#c9c8c3"
GRID_COLOR = "#e6e5e1"

# Index of every class in the label maps
CLASS_INDEX = {"esophagus": 1, "heart": 2, "trachea": 3, "aorta": 4}
FOLDS = [0, 1, 2, 3]

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": LIGHT_GRAY,
    "axes.labelcolor": MUTED_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED_COLOR,
    "ytick.color": MUTED_COLOR,
    "legend.frameon": False,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


def label(method: str) -> str:
    return METHODS[method][0]


def color(method: str) -> str:
    return METHODS[method][1]


def legend_handles(methods: list[str]) -> list:
    return [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=color(m), label=label(m))
            for m in methods]


def read_dice_csv(path: Path) -> tuple[dict[str, dict], list[str], list[str]]:
    # Returns patients[name] = {"contrast": 0 or 1, method: {class: dice}}, the methods and the classes
    patients: dict[str, dict] = {}
    methods: list[str] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        classes: list[str] = [c for c in reader.fieldnames if c not in ["patient", "contrast", "method"]]
        for row in reader:
            name = row["patient"]
            if name not in patients:
                patients[name] = {"contrast": int(row["contrast"])}
            if row["method"] not in methods:
                methods.append(row["method"])
            patients[name][row["method"]] = {c: float(row[c]) for c in classes}
    return patients, methods, classes


def find_fold(patient: str, data_dir: str) -> int:
    # The fold in which this patient was in the validation set
    for fold in FOLDS:
        if list(Path(f"{data_dir}_fold{fold}/val/img").glob(f"{patient}_*.png")):
            return fold
    raise ValueError(patient)


def plot_input_examples(dest: Path, data_prefix: str, methods: list[str],
                        patients: list[str], contrast: dict[str, int]) -> None:
    fig, axes = plt.subplots(len(patients), len(methods), figsize=(3.2 * len(methods), 3.4 * len(patients)),
                             squeeze=False)

    for row, patient in enumerate(patients):
        fold = find_fold(patient, f"{data_prefix}_{methods[0]}")

        # Take the slice with the most aorta pixels (aorta = 252 in the .png)
        gt_paths = sorted(Path(f"{data_prefix}_{methods[0]}_fold{fold}/val/gt").glob(f"{patient}_*.png"))
        aorta_sizes = [(np.array(Image.open(p)) == 252).sum() for p in gt_paths]
        slice_name = gt_paths[int(np.argmax(aorta_sizes))].name

        for col, method in enumerate(methods):
            data_name = DATA_OF[method] if method in DATA_OF else method
            img = np.array(Image.open(f"{data_prefix}_{data_name}_fold{fold}/val/img/{slice_name}"))
            ax = axes[row, col]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(img)  # 3 windows shown as the red, green and blue channels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if row == 0:
                title = label(method)
                if method == "multi":
                    title += "\nR: -150..250, G: -1000..-200,\nB: 100..500 HU"
                ax.set_title(title, fontsize=9)
            if col == 0:
                group = "contrast-enhanced" if contrast[patient] == 1 else "non-enhanced"
                ax.set_ylabel(f"{patient}\n{group}", fontsize=9, color=TEXT_COLOR)

    fig.savefig(dest)
    plt.close(fig)


def plot_dice_change(dest: Path, patients: dict[str, dict], methods: list[str], classes: list[str]) -> None:
    # One panel per class. Every dot is one patient: its Dice with a method minus its Dice with
    # the baseline. Above 0 = better than the baseline.
    reference = methods[0]
    others = methods[1:]
    groups = [("Contrast-\nenhanced", 1), ("Non-\nenhanced", 0)]
    rng = np.random.default_rng(0)  # Small horizontal jitter so the dots do not hide each other

    fig, axes = plt.subplots(1, len(classes), figsize=(3.0 * len(classes), 4.2), sharey=True)
    for ax, cls in zip(axes, classes):
        ticks, tick_labels = [], []
        for g, (group_name, is_contrast) in enumerate(groups):
            names = [p for p in patients if patients[p]["contrast"] == is_contrast]
            for m, method in enumerate(others):
                x = g * (len(others) + 1) + m
                change = np.array([patients[p][method][cls] - patients[p][reference][cls] for p in names])
                ax.scatter(x + rng.uniform(-0.2, 0.2, len(change)), change, s=16, color=color(method),
                           edgecolor="white", linewidth=0.6, zorder=2)
                median = np.median(change)
                ax.plot([x - 0.35, x + 0.35], [median, median], color=TEXT_COLOR, linewidth=2, zorder=3)
            ticks.append(g * (len(others) + 1) + (len(others) - 1) / 2)
            tick_labels.append(f"{group_name}\n(n={len(names)})")

        ax.axhline(0, color=MUTED_COLOR, linewidth=0.8, zorder=1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=8.5)
        ax.set_title(cls if cls != "mean" else "mean of the classes")
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel(f"3D Dice change vs {label(reference).lower()}")
    handles = legend_handles(others) + [plt.Line2D([], [], color=TEXT_COLOR, linewidth=2, label="Group median")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.05))
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(dest)
    plt.close(fig)


def plot_dice_per_patient(dest: Path, patients: dict[str, dict], methods: list[str]) -> None:
    # One row per patient, one dot per method. Contrast-enhanced patients on top, then
    # non-enhanced, each sorted by the baseline Dice.
    reference = methods[0]
    rows: list[str] = []
    for is_contrast in [1, 0]:
        names = [p for p in patients if patients[p]["contrast"] == is_contrast]
        names.sort(key=lambda p: patients[p][reference]["mean"], reverse=True)
        rows += names
    n_contrast = sum(1 for p in patients if patients[p]["contrast"] == 1)

    fig, ax = plt.subplots(figsize=(7, 0.26 * len(rows) + 1.5))
    for y, p in enumerate(rows):
        values = [patients[p][m]["mean"] for m in methods]
        ax.plot([min(values), max(values)], [y, y], color=LIGHT_GRAY, linewidth=1.5, zorder=1)
        for method, value in zip(methods, values):
            ax.scatter([value], [y], s=36, color=color(method), edgecolor="white", linewidth=1, zorder=2)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([p.replace("Patient_", "Patient ") for p in rows], fontsize=8)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)

    # Separate the two groups
    ax.axhline(n_contrast - 0.5, color=MUTED_COLOR, linewidth=0.8)
    ax.text(1.01, (n_contrast - 1) / 2, "Contrast-\nenhanced", va="center", fontsize=9,
            color=MUTED_COLOR, transform=ax.get_yaxis_transform())
    ax.text(1.01, (n_contrast + len(rows) - 1) / 2, "Non-\nenhanced", va="center", fontsize=9,
            color=MUTED_COLOR, transform=ax.get_yaxis_transform())

    ax.set_xlabel("Mean 3D Dice over the classes")
    ax.set_title("Per-patient Dice for every method", fontsize=10)
    ax.legend(handles=legend_handles(methods), loc="upper center", bbox_to_anchor=(0.5, -2.0 / len(rows)), ncol=2)
    fig.savefig(dest)
    plt.close(fig)


def plot_contrast_gap(dest: Path, patients: dict[str, dict], methods: list[str], classes: list[str]) -> None:
    # Gap = median Dice of the contrast-enhanced patients - median Dice of the non-enhanced ones.
    # Positive: the model does better on contrast-enhanced scans.
    fig, ax = plt.subplots(figsize=(7, 0.9 * len(classes) + 1.2))
    bar_height = 0.8 / len(methods)
    y = np.arange(len(classes))

    for i, method in enumerate(methods):
        gaps = []
        for cls in classes:
            enhanced = [patients[p][method][cls] for p in patients if patients[p]["contrast"] == 1]
            non_enhanced = [patients[p][method][cls] for p in patients if patients[p]["contrast"] == 0]
            gaps.append(np.median(enhanced) - np.median(non_enhanced))

        # The bars of one class are next to each other, with a small gap between them
        y_pos = y - 0.4 + bar_height * (i + 0.5)
        ax.barh(y_pos, gaps, height=bar_height * 0.85, color=color(method), label=label(method))
        for yy, gap in zip(y_pos, gaps):
            ax.text(gap + (0.003 if gap >= 0 else -0.003), yy, f"{gap:+.3f}", va="center",
                    ha="left" if gap >= 0 else "right", fontsize=7.5, color=MUTED_COLOR)

    ax.axvline(0, color=MUTED_COLOR, linewidth=0.8)
    limit = max(abs(v) for v in ax.get_xlim()) + 0.02
    ax.set_xlim(-limit, limit)
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Median Dice: enhanced − non-enhanced")
    ax.set_title("Gap between contrast-enhanced and non-enhanced patients\n"
                 "(> 0: enhanced scans are segmented better)", fontsize=10)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8.5)
    fig.savefig(dest)
    plt.close(fig)


def plot_training_curves(dest: Path, results_dir: Path, methods: list[str], classes: list[str]) -> None:
    # Validation Dice during training (2D, averaged over all validation slices, as in main.py).
    # First row: mean over the classes. Then one row per small organ, which are the hardest to learn.
    organ_classes = [c for c in classes if c in CLASS_INDEX]
    rows = [("mean", [CLASS_INDEX[c] for c in organ_classes])]
    for small_organ in ["esophagus", "trachea"]:
        if small_organ in organ_classes:
            rows.append((small_organ, [CLASS_INDEX[small_organ]]))

    fig, axes = plt.subplots(len(rows), len(FOLDS), figsize=(13, 2.6 * len(rows) + 0.6),
                             sharex=True, sharey="row", squeeze=False)
    for col, fold in enumerate(FOLDS):
        for method in methods:
            dice = np.load(results_dir / f"{method}_fold{fold}" / "dice_val.npy")  # (epochs, slices, K)
            epochs = np.arange(1, dice.shape[0] + 1)
            for row, (_, class_ids) in enumerate(rows):
                axes[row, col].plot(epochs, dice[:, :, class_ids].mean(axis=(1, 2)), color=color(method),
                                    linewidth=2, label=label(method))
        axes[0, col].set_title(f"Fold {fold}")
        axes[-1, col].set_xlabel("Epoch")

    for row, (row_name, _) in enumerate(rows):
        axes[row, 0].set_ylabel(f"Val Dice ({row_name})")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Validation Dice during training (2D, per slice; empty slices count as 1, "
                 "so a flat line = organ never predicted)", fontsize=10, y=1.0)
    fig.savefig(dest)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    dice_csv = args.dice_csv if args.dice_csv is not None else args.results_dir / "dice_3d.csv"
    patients, methods, classes = read_dice_csv(dice_csv)
    contrast = {p: patients[p]["contrast"] for p in patients}
    print(f"{len(patients)} patients, methods {methods}, classes {classes}")

    dest_dir = args.dest_dir if args.dest_dir is not None else args.results_dir / "figures"
    dest_dir.mkdir(parents=True, exist_ok=True)

    plot_input_examples(dest_dir / "1_input_examples.png", args.data_prefix, methods,
                        args.example_patients, contrast)
    plot_dice_change(dest_dir / "2_dice_change.png", patients, methods, classes)
    plot_dice_per_patient(dest_dir / "3_dice_per_patient.png", patients, methods)
    plot_contrast_gap(dest_dir / "4_contrast_gap.png", patients, methods, classes)
    plot_training_curves(dest_dir / "5_training_curves.png", args.results_dir, methods, classes)

    print(f"Saved the figures to {dest_dir}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the results of the HU window experiment")
    parser.add_argument('--results_dir', type=Path, default=Path("results/hu_window_full"),
                        help="Folder written by run_all.sh (with dice_3d.csv and the <method>_fold<k> runs)")
    parser.add_argument('--dice_csv', type=Path, default=None,
                        help="Output of evaluate_contrast.py (default: <results_dir>/dice_3d.csv)")
    parser.add_argument('--dest_dir', type=Path, default=None,
                        help="Where to save the charts (default: <results_dir>/figures)")
    parser.add_argument('--data_prefix', type=str, default="data/SEGTHOR_FULL",
                        help="Prefix of the sliced data: <data_prefix>_<method>_fold<k>")
    parser.add_argument('--example_patients', nargs='+', default=["Patient_31", "Patient_19"],
                        help="Patients shown in 1_input_examples.png (default: the enhanced patient "
                             "whose aorta is the brightest, and the scan with the brightest metal)")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
