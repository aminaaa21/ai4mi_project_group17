"""Run from the repository root: python error_analysis.py
Outputs are saved in results/error_analysis/.
"""
from pathlib import Path
import csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

GT_ROOT = Path("data/train")
PRED_ROOT = Path("volumes/single_changes/windowing")
METRIC_ROOT = Path("results/METRICS/single_changes/windowing")
OUT = Path("results/ERROR_ANALYSIS/single_changes/windowing")
OUT.mkdir(parents=True, exist_ok=True)
LABELS = (1, 2, 3, 4)
CLASS_NAMES = {1: "Esophagus", 2: "Heart", 3:"Trachea", 4:"Aorta"}
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
EXAMPLES = [("Patient_35", 1), ("Patient_22", 3),
            ("Patient_30", 2), ("Patient_21", 3)]
CT_WINDOW = (-160, 240)
METRICS = ("dice", "hd95_mm", "msd_mm")
TITLES = ("Dice — higher is better", "HD95 (mm) — lower is better",
          "MSD (mm) — lower is better")


def read_metrics(name):
    with np.load(METRIC_ROOT / f"{name}.npz", allow_pickle=False) as archive:
        return {k: np.asarray(archive[k], dtype=float).copy() for k in archive.files}


def load_mask(path):
    image = nib.load(str(path))
    raw = np.asarray(image.dataobj)
    if raw.ndim != 3 or not np.all(np.isin(raw, [0, 1, 2, 3, 4])):
        raise ValueError(f"Expected a 3D mask with labels 0–4: {path}")
    return image, raw.astype(np.uint8)


def check_grid(reference, other, description):
    if reference.shape != other.shape:
        raise ValueError(f"Shape mismatch: {description}")
    if not np.allclose(reference.affine, other.affine, rtol=1e-5, atol=1e-5):
        raise ValueError(f"Affine mismatch: {description}")


def write_csv(filename, rows):
    if rows:
        with (OUT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def mean_sd(values):
    array = np.asarray(values, dtype=float)
    if np.all(np.isfinite(array)):
        return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0
    return float(np.mean(array)), float("nan")


def plot_metrics_per_class(rows, patients, class_summary):
    """Four separate bars per patient"""
    lookup = {(r["patient"], r["label"]): r for r in rows}
    x = np.arange(len(patients))
    width = 0.19
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), constrained_layout=True)
    for ax, metric, title in zip(axes, METRICS, TITLES):
        for i, label in enumerate(LABELS):
            values = np.array([lookup[(p, label)][metric] for p in patients])
            positions = x + (i - 1.5) * width
            # Nonfinite values cannot be meaningfully drawn as ordinary bars.
            finite = np.isfinite(values)
            ax.bar(positions[finite], values[finite], width=width,
                   color=COLORS[i], label=CLASS_NAMES[label])
            for position in positions[~finite]:
                ax.text(position, 0.03, "NA/inf", rotation=90, ha="center",
                        va="bottom", fontsize=7, transform=ax.get_xaxis_transform())
        ax.set_xticks(x)
        ax.set_xticklabels(patients, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Dice" if metric == "dice" else "mm")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(ncol=4, fontsize=9)
        if metric == "dice":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=0)
    fig.suptitle("Per-patient, per-class metrics — background excluded", fontsize=16)
    fig.savefig(OUT / "metrics_per_patient_per_class.png", dpi=160)
    plt.close(fig)

    # One bar per class: average over patients only, not over classes.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, metric, title in zip(axes, METRICS, TITLES):
        values = [r[f"{metric}_mean"] for r in class_summary]
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.bar(i, value, color=COLORS[i], width=0.65)
                ax.annotate(f"{value:.3f}" if metric == "dice" else f"{value:.2f}",
                            (i, value), xytext=(0, 5), textcoords="offset points",
                            ha="center", fontsize=9)
            else:
                ax.text(i, 0.05, "NA/inf", ha="center",
                        transform=ax.get_xaxis_transform())
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels([CLASS_NAMES[c] for c in LABELS], rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Dice" if metric == "dice" else "mm")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        if metric == "dice":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=0)
            ax.margins(y=0.18)
    fig.suptitle("Per-class means across validation patients — background excluded")
    fig.savefig(OUT / "metrics_mean_per_class.png", dpi=160)
    plt.close(fig)

    # Wide table: every patient's four class values are visible side by side.
    wide = []
    for patient in patients:
        record = {"patient": patient}
        for metric in METRICS:
            for label in LABELS:
                record[f"{metric}_label_{label}"] = lookup[(patient, label)][metric]
        wide.append(record)
    write_csv("metrics_per_patient_per_class_wide.csv", wide)


def plot_example(patient, label):
    source = GT_ROOT / patient
    ct_img = nib.load(str(source / f"{patient}.nii.gz"))
    gt_img = nib.load(str(source / "GT.nii.gz"))
    pred_img = nib.load(str(PRED_ROOT / f"{patient}.nii.gz"))
    check_grid(gt_img, ct_img, f"{patient}: CT versus GT")
    check_grid(gt_img, pred_img, f"{patient}: prediction versus GT")
    # Canonical third axis is inferior-superior. Oblique scans are not resampled.
    ct_img = nib.as_closest_canonical(ct_img)
    gt_img = nib.as_closest_canonical(gt_img)
    pred_img = nib.as_closest_canonical(pred_img)
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    gt = np.asarray(gt_img.dataobj) == label
    pred = np.asarray(pred_img.dataobj) == label
    errors = np.count_nonzero(gt != pred, axis=(0, 1))
    gt_area = np.count_nonzero(gt, axis=(0, 1))
    selections = []
    if errors.max() > 0:
        selections.append(("largest_error", int(errors.argmax())))
    if gt_area.max() > 0:
        z = int(gt_area.argmax())
        if not any(old_z == z for _, old_z in selections):
            selections.append(("largest_gt_area", z))
    if not selections:
        print(f"Skipping absent label: {patient}, {label}")
        return
    cmap = ListedColormap(["#2ca02c", "#ff7f0e", "#d62728"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    spacing = ct_img.header.get_zooms()[:3]
    extent = (0, ct.shape[0] * spacing[0], 0, ct.shape[1] * spacing[1])
    for selection, z in selections:
        g, p = gt[:, :, z], pred[:, :, z]
        tp, fp, fn = g & p, ~g & p, g & ~p
        error_map = np.zeros(g.shape, dtype=np.uint8)
        error_map[tp], error_map[fp], error_map[fn] = 1, 2, 3
        fig, axes = plt.subplots(1, 4, figsize=(17, 5), constrained_layout=True)
        for ax in axes:
            ax.imshow(ct[:, :, z].T, cmap="gray", origin="lower",
                      vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], extent=extent)
            ax.set_aspect("equal")
            ax.set_axis_off()
        for ax, mask, color in [(axes[1], g, "#00bfff"), (axes[2], p, "#ff00ff")]:
            ax.imshow(np.ma.masked_where(~mask.T, mask.T),
                      cmap=ListedColormap([color]), vmin=0, vmax=1,
                      alpha=0.45, origin="lower", extent=extent)
        axes[3].imshow(np.ma.masked_where(error_map.T == 0, error_map.T),
                       cmap=cmap, norm=norm, alpha=0.65, origin="lower", extent=extent)
        for ax, title in zip(axes, ["CT", "Ground truth", "Prediction", "TP / FP / FN"]):
            ax.set_title(title)
        axes[3].legend(handles=[Patch(color="#2ca02c", label="TP"),
                                Patch(color="#ff7f0e", label="FP"),
                                Patch(color="#d62728", label="FN")], loc="lower right")
        denominator = int(2 * tp.sum() + fp.sum() + fn.sum())
        dice = 2 * int(tp.sum()) / denominator if denominator else float("nan")
        fig.suptitle(f"{patient} | {CLASS_NAMES[label]} | {selection} | "
                     f"canonical slice {z} | slice Dice {dice:.3f}\n"
                     f"TP {tp.sum():,} | FP {fp.sum():,} | FN {fn.sum():,}")
        path = OUT / f"{patient}_label{label}_{selection}_z{z}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)


def main():
    metrics = {name: read_metrics(name) for name in ("dice", "hd95", "msd")}
    patients = sorted(metrics["dice"])
    if not patients:
        raise ValueError("No patients found in saved metrics")
    for name, values in metrics.items():
        if set(values) != set(patients):
            raise ValueError(f"Patient keys differ in {name}.npz")
        for patient in patients:
            if values[patient].shape != (5,):
                raise ValueError(f"Expected 5 class entries: {name}, {patient}")
    rows = []
    for patient in patients:
        gt_img, gt = load_mask(GT_ROOT / patient / "GT.nii.gz")
        pred_img, pred = load_mask(PRED_ROOT / f"{patient}.nii.gz")
        check_grid(gt_img, pred_img, f"{patient}: prediction versus GT")
        voxel_ml = abs(float(np.linalg.det(gt_img.affine[:3, :3]))) / 1000
        for label in LABELS:
            g, p = gt == label, pred == label
            tp = int(np.count_nonzero(g & p))
            fp = int(np.count_nonzero(~g & p))
            fn = int(np.count_nonzero(g & ~p))
            gt_n, pred_n = tp + fn, tp + fp
            denominator = 2 * tp + fp + fn
            dice = 2 * tp / denominator if denominator else float("nan")
            saved = float(metrics["dice"][patient][label])
            if denominator and not np.isclose(dice, saved, atol=1e-4, rtol=1e-4):
                raise ValueError(f"Dice mismatch: {patient}, label {label}: "
                                 f"recomputed={dice:.6f}, saved={saved:.6f}. "
                                 "Check array label indexing and mask/metric provenance.")
            rows.append({
                "patient": patient, "label": label, "class_name": CLASS_NAMES[label],
                "dice": dice, "hd95_mm": float(metrics["hd95"][patient][label]),
                "msd_mm": float(metrics["msd"][patient][label]),
                "tp_voxels": tp, "fp_voxels": fp, "fn_voxels": fn,
                "precision": tp / pred_n if pred_n else float("nan"),
                "recall": tp / gt_n if gt_n else float("nan"),
                "gt_volume_ml": gt_n * voxel_ml, "pred_volume_ml": pred_n * voxel_ml,
                "fp_volume_ml": fp * voxel_ml, "fn_volume_ml": fn * voxel_ml,
                "signed_volume_error_percent": 100 * (pred_n - gt_n) / gt_n if gt_n else float("nan")})
            del g, p
        del gt, pred
    write_csv("per_patient_per_class.csv", rows)
    class_summary = []
    for label in LABELS:
        subset = [r for r in rows if r["label"] == label]
        summary = {"label": label, "class_name": CLASS_NAMES[label], "n_patients": len(subset)}
        for name in (*METRICS, "precision", "recall"):
            values = [r[name] for r in subset]
            summary[f"{name}_mean"], summary[f"{name}_sd"] = mean_sd(values)
            summary[f"{name}_median"] = float(np.median(values))
            summary[f"{name}_n_nonfinite"] = int(np.count_nonzero(~np.isfinite(values)))
        class_summary.append(summary)
    write_csv("per_class_summary.csv", class_summary)
    print("\nPER-CLASS RESULTS (mean across patients):")
    print(f"{'Class':<16} {'Dice':>9} {'HD95 (mm)':>12} {'MSD (mm)':>12}")
    for r in class_summary:
        print(f"{r['class_name']:<16} {r['dice_mean']:>9.3f} "
              f"{r['hd95_mm_mean']:>12.2f} {r['msd_mm_mean']:>12.2f}")
    patient_summary = []
    for patient in patients:
        subset = [r for r in rows if r["patient"] == patient]
        patient_summary.append({"patient": patient, **{
            f"mean_foreground_{name}": float(np.mean([r[name] for r in subset]))
            for name in METRICS}})
    write_csv("per_patient_foreground.csv", patient_summary)
    write_csv("ranked_lowest_dice.csv", sorted(rows, key=lambda r: r["dice"]))
    write_csv("ranked_highest_hd95.csv", sorted(rows, key=lambda r: r["hd95_mm"], reverse=True))
    overall = {"n_patients": len(patients), "n_foreground_classes": len(LABELS)}
    for name in METRICS:
        values = [r[f"mean_foreground_{name}"] for r in patient_summary]
        overall[f"{name}_macro_mean"], overall[f"{name}_patient_sd"] = mean_sd(values)
    write_csv("overall_foreground_summary.csv", [overall])
    plot_metrics_per_class(rows, patients, class_summary)
    for patient, label in EXAMPLES:
        if patient in patients:
            plot_example(patient, label)


if __name__ == "__main__":
    main()
