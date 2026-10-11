#!/usr/bin/env python3

# Live progress of the HU window experiment, while hu_window/run_all.sh is still running.
#
# main.py saves dice_val.npy after every epoch, so we can plot the validation Dice of every run
# that has started so far (also the one that is still training), and print a short table.
#
# Note: this is the 2D Dice of main.py (per slice, empty slices count as 1), not the final 3D Dice.

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Same names, colors and chart style as the final charts
from plot_results import METHODS, CLASS_INDEX, FOLDS, label, color


def completed_epochs(dice: np.ndarray) -> int:
    # dice_val.npy is filled with zeros for the epochs that did not run yet
    done = 0
    for e in range(dice.shape[0]):
        if dice[e].sum() > 0:
            done = e + 1
    return done


def main(args: argparse.Namespace) -> None:
    rows = [("mean of the organs", list(CLASS_INDEX.values()))]
    rows += [(name, [index]) for name, index in CLASS_INDEX.items()]

    fig, axes = plt.subplots(len(rows), len(FOLDS), figsize=(13, 2.2 * len(rows) + 0.8),
                             sharex=True, sharey=True, squeeze=False)

    print(f"{'run':<16} {'epochs':>7} {'best val Dice':>14}   (2D, mean of the organs)")
    methods_seen: list[str] = []
    for col, fold in enumerate(FOLDS):
        for method in METHODS:
            dice_file = args.results_dir / f"{method}_fold{fold}" / "dice_val.npy"
            if not dice_file.exists():
                continue
            dice = np.load(dice_file)  # (epochs, slices, K)
            n_done = completed_epochs(dice)
            if n_done == 0:
                continue
            if method not in methods_seen:
                methods_seen.append(method)

            epochs = np.arange(1, n_done + 1)
            for row, (_, class_ids) in enumerate(rows):
                values = dice[:n_done][:, :, class_ids].mean(axis=(1, 2))
                axes[row, col].plot(epochs, values, color=color(method), linewidth=2, label=label(method))

            mean_dice = dice[:n_done][:, :, list(CLASS_INDEX.values())].mean(axis=(1, 2))
            print(f"{method + '_fold' + str(fold):<16} {n_done:>4}/{dice.shape[0]:<2} {mean_dice.max():>14.3f}")

        axes[0, col].set_title(f"Fold {fold}")
        axes[-1, col].set_xlabel("Epoch")
        axes[-1, col].set_xlim(0, 26)
        for row in range(len(rows)):
            axes[row, col].set_ylim(0, 1)  # Same scale for every panel, to compare the runs

    for row, (row_name, _) in enumerate(rows):
        axes[row, 0].set_ylabel(f"Val Dice\n({row_name})")

    handles = [plt.Line2D([], [], color=color(m), linewidth=2, label=label(m)) for m in methods_seen]
    fig.suptitle("Training progress: validation Dice per epoch (2D, per slice; a flat line at the "
                 "start = organ not predicted yet)", fontsize=10, y=0.995)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.975))
    fig.subplots_adjust(top=0.92)

    dest = args.results_dir / "figures" / "progress.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    print(f"Saved {dest}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the training progress of the HU window experiment")
    parser.add_argument('--results_dir', type=Path, default=Path("results/hu_window_full"),
                        help="Folder written by run_all.sh")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
