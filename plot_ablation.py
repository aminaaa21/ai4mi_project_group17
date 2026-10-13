#!/usr/bin/env python3

# Charts for the final presentation:
#
#   ablation_<item>.png      the full model against the full model without one item:
#                            3D Dice, HD95 and MSD per organ (slides 9-11)
#   windowing.png            baseline against 1 window and 3 windows, only windowing changed (slide 12)
#   window_vs_25d.png        1 window + 2.5D against 3 windows, with the same settings (slide 13)
#   ablation_overview.png    how much worse (or better) the full model gets without every item (slide 15)
#   best_vs_baseline.png     our best model against the baseline: 3D Dice, HD95 and MSD per organ (slide 16)
#
# The 3D metrics are read from the output folders of evaluate_3d.py (dice.npz, hd95.npz, msd.npz),
# one folder per run: results/METRICS/<run>, e.g. results/METRICS/leave_one_out/no_windowing.

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # No window, only save the figures
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ORGANS = ["Esophagus", "Heart", "Trachea", "Aorta"]

BASELINE = "baseline"

# The full model of the ablation study: 3 HU windows + augmentation + Dice and weighted cross-entropy +
# weight decay, learning rate 0.0005
FULL_MODEL = "leave_one_out/no_lower_lr"

# Our best model: the full model without weight decay
BEST_MODEL = "leave_one_out/no_weight_decay"

# (name on the slides, file name, run without this item)
ITEMS = [("Dice + weighted CE", "dice_weighted_ce", "ablation_lr0005/no_weighted_loss"),
         ("Augmentation", "augmentation", "ablation_lr0005/no_augmentation"),
         ("Weight decay", "weight_decay", BEST_MODEL),
         ("Windowing", "windowing", "ablation_lr0005/no_windowing")]

REFERENCE_COLOR = "#8f8e89"  # The model we compare with (full model or baseline), in gray
CHANGED_COLOR = "#2a78d6"  # The model with one item left out (or one change added)
SECOND_COLOR = "#eb6834"  # A second changed model, when 3 models are compared
HELPS_COLOR = "#2a78d6"  # The item makes the model better
HURTS_COLOR = "#e34948"  # The item makes the model worse
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e6e5e1"

METRIC_TITLES = {"dice": "3D Dice per organ, higher is better",
                 "hd95": "HD95 per organ (mm), lower is better",
                 "msd": "MSD per organ (mm), lower is better"}


def load_metric(metrics_dir, run, metric):
    # Returns an array (patients, 4): one value per patient and organ (the background is left out)
    values = np.load(metrics_dir / run / f"{metric}.npz")
    return np.array([values[patient][1:5] for patient in sorted(values.keys())])


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED_COLOR)
    ax.spines["bottom"].set_color(MUTED_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=11)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)


def plot_per_organ(metrics_dir, models, metrics, dest, figsize):
    # models: list of (name, run), the first one in gray. One panel per metric, bars per organ + the mean.
    # With 2 models the difference (second - first) is written above every pair of bars.
    colors = [REFERENCE_COLOR, CHANGED_COLOR, SECOND_COLOR]
    labels = ORGANS + ["Mean"]
    x = np.arange(len(labels))
    width = 0.72 / len(models)

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    for ax, metric in zip(axes, metrics):
        values = []
        for name, run in models:
            per_patient = load_metric(metrics_dir, run, metric)
            values.append(list(per_patient.mean(axis=0)) + [per_patient.mean()])

        for m in range(len(models)):
            offset = (m - (len(models) - 1) / 2) * (width + 0.02)
            ax.bar(x + offset, values[m], width, color=colors[m], label=models[m][0])

        largest = max(max(v) for v in values)
        if len(models) == 2:
            for i in range(len(labels)):
                difference = values[1][i] - values[0][i]
                if metric == "dice":
                    number = f"{difference:+.2f}"
                else:
                    number = f"{difference:+.1f}"
                top = max(values[0][i], values[1][i])
                bold = "bold" if labels[i] == "Mean" else "normal"
                ax.text(x[i], top + largest * 0.02, number, ha="center", va="bottom", fontsize=10,
                        color=TEXT_COLOR, fontweight=bold)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12 if len(metrics) == 2 else 9, color=TEXT_COLOR)
        ax.set_title(METRIC_TITLES[metric], fontsize=13, color=TEXT_COLOR, loc="left")
        if metric == "dice":
            ax.set_ylim(0, 1.08)
        else:
            ax.set_ylim(0, largest * 1.15)
        style_axes(ax)

    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=12, loc="upper center", ncol=len(models))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(dest, dpi=200, facecolor="white")
    plt.close(fig)


def plot_window_vs_25d(metrics_dir, pairs, dest):
    # pairs: list of (settings, run with 1 window + 2.5D, run with 3 windows).
    # Left: mean 3D Dice, right: mean HD95, one pair of bars per setting.
    names = ["1 window + 2.5D", "3 windows"]
    colors = [REFERENCE_COLOR, CHANGED_COLOR]
    x = np.arange(len(pairs))
    width = 0.36

    fig, (ax_dice, ax_hd95) = plt.subplots(1, 2, figsize=(10.5, 3.5))
    for ax, metric, title in [(ax_dice, "dice", "Mean 3D Dice, higher is better"),
                              (ax_hd95, "hd95", "Mean HD95 (mm), lower is better")]:
        largest = 0
        for m in range(2):
            values = []
            for settings, run_25d, run_3w in pairs:
                run = run_25d if m == 0 else run_3w
                values.append(np.mean(load_metric(metrics_dir, run, metric)))
            offset = (m - 0.5) * (width + 0.02)
            ax.bar(x + offset, values, width, color=colors[m], label=names[m])
            for i in range(len(values)):
                number = f"{values[i]:.2f}" if metric == "dice" else f"{values[i]:.1f}"
                ax.text(x[i] + offset, values[i] * 1.02, number, ha="center", va="bottom", fontsize=10,
                        color=TEXT_COLOR)
            largest = max(largest, max(values))

        ax.set_xticks(x)
        ax.set_xticklabels([settings for settings, run_25d, run_3w in pairs], fontsize=11, color=TEXT_COLOR)
        ax.set_title(title, fontsize=13, color=TEXT_COLOR, loc="left")
        if metric == "dice":
            ax.set_ylim(0, 1.0)
        else:
            ax.set_ylim(0, largest * 1.2)
        style_axes(ax)

    handles, labels = ax_dice.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=12, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(dest, dpi=200, facecolor="white")
    plt.close(fig)


def plot_changes(names, changes, dest):
    # changes[metric] = list of differences (one per name). For Dice higher is better,
    # for HD95 and MSD lower is better.
    metrics = [("dice", "3D Dice", True), ("hd95", "HD95 (mm)", False), ("msd", "MSD (mm)", False)]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    y = np.arange(len(names))[::-1]  # The first name on top

    for ax, (metric, label, higher_is_better) in zip(axes, metrics):
        values = changes[metric]
        colors = []
        for value in values:
            if higher_is_better:
                better = value > 0
            else:
                better = value < 0
            colors.append(HELPS_COLOR if better else HURTS_COLOR)

        ax.barh(y, values, height=0.6, color=colors)
        ax.axvline(0, color=MUTED_COLOR, linewidth=1)
        limit = max(abs(value) for value in values) * 1.8
        ax.set_xlim(-limit, limit)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))

        # The value next to every bar
        for i in range(len(values)):
            number = f"{values[i]:+.3f}" if metric == "dice" else f"{values[i]:+.1f}"
            if values[i] >= 0:
                ax.text(values[i] + limit * 0.03, y[i], number, va="center", ha="left", fontsize=11,
                        color=TEXT_COLOR)
            else:
                ax.text(values[i] - limit * 0.03, y[i], number, va="center", ha="right", fontsize=11,
                        color=TEXT_COLOR)

        direction = "higher is better" if higher_is_better else "lower is better"
        ax.set_title(f"{label}, {direction}", fontsize=13, color=TEXT_COLOR, loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(MUTED_COLOR)
        ax.tick_params(colors=MUTED_COLOR, labelsize=11)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=1)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=12, color=TEXT_COLOR)

    # Legend for the two colors
    legend = [Patch(color=HELPS_COLOR, label="model got better"), Patch(color=HURTS_COLOR, label="model got worse")]
    fig.legend(handles=legend, frameon=False, fontsize=12, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(dest, dpi=200, facecolor="white")
    plt.close(fig)


def main(args):
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. One chart per item: the full model against the full model without that item
    names = []
    changes = {"dice": [], "hd95": [], "msd": []}
    for item, file_name, run in ITEMS:
        models = [("Full model", FULL_MODEL), (f"No {item}", run)]
        dest = args.dest_dir / f"ablation_{file_name}.png"
        plot_per_organ(args.metrics_dir, models, ("dice", "hd95", "msd"), dest, figsize=(12, 4))
        print(f"Saved {dest}")

        # Leaving the item out changes the mean metric by (without - full)
        names.append(f"No {item}")
        for metric in changes:
            without = load_metric(args.metrics_dir, run, metric)
            full = load_metric(args.metrics_dir, FULL_MODEL, metric)
            changes[metric].append(np.mean(without) - np.mean(full))

    # 2. Overview: what happens to the full model without every item
    dest = args.dest_dir / "ablation_overview.png"
    plot_changes(names, changes, dest)
    print(f"Saved {dest}")

    # 3. Windowing: baseline against 1 window and 3 windows, nothing else changed
    models = [("Baseline (min-max)", BASELINE),
              ("1 window (soft tissue)", "single_changes/windowing_1ch"),
              ("3 windows", "single_changes/windowing")]
    dest = args.dest_dir / "windowing.png"
    plot_per_organ(args.metrics_dir, models, ("dice", "hd95"), dest, figsize=(10.5, 3.5))
    print(f"Saved {dest}")

    # 4. 1 window + 2.5D against 3 windows (they can not be combined), with the same settings
    pairs = [("lr 0.00005\n+ weight decay", "full_model_25d/full", "full_model"),
             ("lr 0.0005\n+ weight decay", "full_model_25d/no_lower_lr", FULL_MODEL),
             ("lr 0.0005\nno weight decay", "full_model_25d/no_weight_decay", BEST_MODEL)]
    dest = args.dest_dir / "window_vs_25d.png"
    plot_window_vs_25d(args.metrics_dir, pairs, dest)
    print(f"Saved {dest}")

    # 5. Our best model against the baseline
    models = [("Baseline", BASELINE), ("Final model", BEST_MODEL)]
    dest = args.dest_dir / "best_vs_baseline.png"
    plot_per_organ(args.metrics_dir, models, ("dice", "hd95", "msd"), dest, figsize=(14, 4.4))
    print(f"Saved {dest}")


def get_args():
    parser = argparse.ArgumentParser(description="Charts for the final presentation")
    parser.add_argument("--metrics_dir", type=Path, default=Path("results/METRICS"),
                        help="Folder with one evaluate_3d.py output folder per run")
    parser.add_argument("--dest_dir", type=Path, default=Path("results/presentation_figures"))
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
