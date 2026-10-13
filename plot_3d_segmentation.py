#!/usr/bin/env python3

# 3D picture of the segmentation of one patient, for the slides:
# the ground truth, the baseline and our final model side by side.
#
# Every voxel of an organ is drawn as a small cube with matplotlib (no smoothing),
# so the picture shows the predictions exactly as they are.
#
# With --organ only that organ is drawn, and the titles show its Dice, HD95 and MSD
# (computed with the functions of evaluate_3d.py) and in how many separate pieces it is.
#
# Examples:
#   python plot_3d_segmentation.py --patient Patient_35 --dest seg_3d_Patient_35.png
#   python plot_3d_segmentation.py --patient Patient_30 --organ aorta --dest seg_3d_Patient_30_aorta.png

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # No window, only save the figure
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import label as find_pieces

from evaluate_3d import surface_distances, hd95_from_distances, msd_from_distances

ORGANS = ["Esophagus", "Heart", "Trachea", "Aorta"]  # Labels 1 to 4
COLORS = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]


def load_volume(path):
    image = nib.load(str(path))
    volume = np.asarray(image.dataobj).astype(np.uint8)
    spacing = image.header.get_zooms()[:3]
    return volume, spacing


def dice(gt, pred, label):
    a = gt == label
    b = pred == label
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return 2 * (a & b).sum() / total


def visible_voxels(mask, axis, direction):
    # The voxels of the mask whose neighbour on one side (one step along the axis, in the
    # direction -1 or +1) is not in the mask: on that side, the voxel is visible from outside.
    neighbour = np.roll(mask, -direction, axis=axis)
    visible = mask & ~neighbour
    return np.argwhere(visible) - 1  # Undo the padding (see organ_voxel_faces)


def add_squares(all_squares, all_colors, voxels, corners, spacing, color):
    # Adds one square per voxel. The corners are given as 0/1 offsets from the corner of the voxel.
    squares = (voxels[:, None, :] + np.array(corners)[None, :, :]) * np.array(spacing)  # In mm
    all_squares.append(squares)
    all_colors += [color] * len(squares)


def organ_voxel_faces(volume, spacing):
    # Draws every voxel as a cube, exactly as it is in the volume (no smoothing).
    # Only the sides of a voxel that touch a voxel outside the organ are visible, so only those
    # are kept. Returns these squares (in mm) and one color per square, for all organs together.
    # All organs go in one list, so matplotlib sorts the squares of different organs correctly.
    # Each side gets its own brightness, so the sides of a cube can be told apart.
    all_squares = []
    all_colors = []
    for label in range(1, 5):
        mask = volume == label
        if mask.sum() == 0:
            continue
        # Pad with zeros, so voxels at the border of the scan also get their outer sides
        mask = np.pad(mask, 1)
        color = np.array(to_rgb(COLORS[label - 1]))

        # Top side (z + 1): brightest
        voxels = visible_voxels(mask, axis=2, direction=+1)
        add_squares(all_squares, all_colors, voxels, [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
                    spacing, color * 1.00)

        # Bottom side (z - 1): darkest
        voxels = visible_voxels(mask, axis=2, direction=-1)
        add_squares(all_squares, all_colors, voxels, [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                    spacing, color * 0.45)

        # Front side (y - 1)
        voxels = visible_voxels(mask, axis=1, direction=-1)
        add_squares(all_squares, all_colors, voxels, [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
                    spacing, color * 0.82)

        # Back side (y + 1)
        voxels = visible_voxels(mask, axis=1, direction=+1)
        add_squares(all_squares, all_colors, voxels, [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
                    spacing, color * 0.82)

        # Left side (x - 1)
        voxels = visible_voxels(mask, axis=0, direction=-1)
        add_squares(all_squares, all_colors, voxels, [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
                    spacing, color * 0.60)

        # Right side (x + 1)
        voxels = visible_voxels(mask, axis=0, direction=+1)
        add_squares(all_squares, all_colors, voxels, [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
                    spacing, color * 0.60)

    return np.concatenate(all_squares), np.array(all_colors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="Patient_35")
    parser.add_argument("--gt_dir", default="data/segthor_train_full/train")
    parser.add_argument("--baseline_dir", default="volumes/segthor/ce")
    parser.add_argument("--ours_dir", default="volumes/leave_one_out/no_weight_decay")
    parser.add_argument("--dest", default="seg_3d.png")
    parser.add_argument("--elev", type=float, default=12)
    parser.add_argument("--azim", type=float, default=-35)
    parser.add_argument("--zoom", type=float, default=1.2, help="Zoom of the 3D plots (1 = matplotlib default)")
    parser.add_argument("--organ", default="all", choices=["all", "esophagus", "heart", "trachea", "aorta"],
                        help="Draw only this organ (default: all organs)")
    args = parser.parse_args()

    gt, spacing = load_volume(Path(args.gt_dir) / args.patient / "GT.nii.gz")
    baseline, _ = load_volume(Path(args.baseline_dir) / f"{args.patient}.nii.gz")
    ours, _ = load_volume(Path(args.ours_dir) / f"{args.patient}.nii.gz")

    # Only one organ: remove the other organs from the three volumes
    if args.organ != "all":
        organ_label = ORGANS.index(args.organ.capitalize()) + 1
        gt = np.where(gt == organ_label, gt, 0)
        baseline = np.where(baseline == organ_label, baseline, 0)
        ours = np.where(ours == organ_label, ours, 0)

    # Crop all three volumes to the box around the organs (in any of the three), plus a margin
    any_organ = (gt > 0) | (baseline > 0) | (ours > 0)
    xs, ys, zs = np.nonzero(any_organ)
    margin = 5
    x0, x1 = max(xs.min() - margin, 0), xs.max() + margin
    y0, y1 = max(ys.min() - margin, 0), ys.max() + margin
    z0, z1 = max(zs.min() - margin, 0), zs.max() + margin
    gt = gt[x0:x1, y0:y1, z0:z1]
    baseline = baseline[x0:x1, y0:y1, z0:z1]
    ours = ours[x0:x1, y0:y1, z0:z1]

    panels = [("Ground truth", gt), ("Baseline", baseline), ("Ours (final model)", ours)]
    size_mm = np.array(gt.shape) * np.array(spacing)

    fig = plt.figure(figsize=(15, 7))
    for i, (title, volume) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d", proj_type="ortho")
        squares, colors = organ_voxel_faces(volume, spacing)
        mesh = Poly3DCollection(squares, facecolors=colors, edgecolor="none")
        mesh.set_clip_on(False)  # Otherwise pieces near the top of the box can be cut off
        ax.add_collection3d(mesh)

        ax.set_xlim(0, size_mm[0])
        ax.set_ylim(0, size_mm[1])
        ax.set_zlim(0, size_mm[2])
        ax.set_box_aspect(size_mm.copy(), zoom=args.zoom)  # set_box_aspect changes the array it gets, so give it a copy
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_axis_off()

        if args.organ == "all":
            if volume is gt:
                ax.set_title(title, fontsize=16, pad=30)
            else:
                scores = [dice(gt, volume, label) for label in range(1, 5)]
                ax.set_title(f"{title}\nmean 3D Dice {np.mean(scores):.3f}", fontsize=16, pad=30)
                print(title, "Dice per organ:", [round(float(s), 3) for s in scores])
        else:
            _, n_pieces = find_pieces(volume == organ_label)
            if n_pieces == 1:
                pieces_text = "1 piece"
            else:
                pieces_text = f"{n_pieces} pieces"
            if volume is gt:
                ax.set_title(f"{title}\n{pieces_text}", fontsize=16, pad=30)
            else:
                organ_dice = dice(gt, volume, organ_label)
                distances = surface_distances(gt == organ_label, volume == organ_label, spacing)
                hd95 = hd95_from_distances(distances)
                msd = msd_from_distances(distances)
                ax.set_title(f"{title}\n{pieces_text}\nDice {organ_dice:.3f}   HD95 {hd95:.1f} mm   MSD {msd:.1f} mm",
                             fontsize=15, pad=30)
                print(title, args.organ, "Dice", round(float(organ_dice), 3), "HD95", round(float(hd95), 1),
                      "MSD", round(float(msd), 1), "pieces", n_pieces)

    if args.organ == "all":
        legend_colors = COLORS
        legend_names = ORGANS
    else:
        legend_colors = [COLORS[organ_label - 1]]
        legend_names = [ORGANS[organ_label - 1]]
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=12, color=c) for c in legend_colors]
    fig.legend(handles, legend_names, loc="lower center", ncol=4, fontsize=14, frameon=False)
    fig.suptitle(args.patient.replace("_", " "), fontsize=18, y=1.02)
    plt.subplots_adjust(left=0, right=1, top=0.8, bottom=0.1, wspace=0)
    fig.savefig(args.dest, dpi=150, bbox_inches="tight")
    print("Saved", args.dest)


if __name__ == "__main__":
    main()
