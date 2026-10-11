import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Load results directory

results_dir = Path("results/baseline_clean_100ep_cpu")

loss_tra = np.load(results_dir / "loss_tra.npy") # Shape: (epochs, batches)
loss_val = np.load(results_dir / "loss_val.npy")
dice_tra = np.load(results_dir / "dice_tra.npy") # Shape: (epochs, samples, K)
dice_val = np.load(results_dir / "dice_val.npy")

# Compute overall mean loss per epoch
mean_loss_tra = loss_tra.mean(axis=1)
mean_loss_val = loss_val.mean(axis=1)

# Compute mean Dice per epoch for each class (Shape: epochs, K)
class_dice_tra = dice_tra.mean(axis=1)
class_dice_val = dice_val.mean(axis=1)

# Compute overall mean Dice across all foreground classes (skipping background class 0)
mean_dice_tra_all = class_dice_tra[:, 1:].mean(axis=1)
mean_dice_val_all = class_dice_val[:, 1:].mean(axis=1)

num_classes = dice_tra.shape[2]
total_epochs = len(mean_loss_tra)
epochs = list(range(1, total_epochs + 1))

# Color coordination palette 
color_palette = mpl.colormaps['tab10'].resampled(num_classes)

def plot_metrics(curr_epochs, l_tra, l_val, d_tra, d_val, d_tra_all, d_val_all, file_suffix, title_suffix, is_window):
    marker_opt = 'o' if is_window else None
    val_marker_opt = 's' if is_window else None

    # --- 1. Overall Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(curr_epochs, l_tra, label="Train Loss", color="blue", marker=marker_opt)
    plt.plot(curr_epochs, l_val, label="Val Loss", color="orange", marker=marker_opt)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Overall Loss - {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f"loss_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. Overall aggregate Dice plot (Train vs Val singular aggregate line) ---
    plt.figure(figsize=(8, 5))
    plt.plot(curr_epochs, d_tra_all, label="Train Mean Dice (All Classes)", color="blue", linestyle='-', linewidth=3, marker=marker_opt)
    plt.plot(curr_epochs, d_val_all, label="Val Mean Dice (All Classes)", color="orange", linestyle='--', linewidth=3, marker=val_marker_opt)
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.title(f"Overall Aggregate Dice - {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f"dice_overall_aggregate_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 3. Train vs Val Dice comparison (All classes) ---
    plt.figure(figsize=(10, 6))
    for c in range(1, num_classes):
        cc = color_palette(c)
        plt.plot(curr_epochs, d_tra[:, c], color=cc, linestyle='-', marker=marker_opt, alpha=0.6, label=f"Train Class {c}")
        plt.plot(curr_epochs, d_val[:, c], color=cc, linestyle='--', marker=val_marker_opt, alpha=0.6, label=f"Val Class {c}")
    plt.plot(curr_epochs, d_tra_all, color='black', linestyle='-', linewidth=3, marker=marker_opt, label="Train Mean (All)")
    plt.plot(curr_epochs, d_val_all, color='black', linestyle='--', linewidth=3, marker=val_marker_opt, label="Val Mean (All)")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.title(f"Train vs Val Dice Comparison - {title_suffix}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f"dice_train_val_comparison_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 4. Training Dice Only ---
    plt.figure(figsize=(10, 6))
    for c in range(1, num_classes):
        cc = color_palette(c)
        plt.plot(curr_epochs, d_tra[:, c], color=cc, linestyle='-', marker=marker_opt, label=f"Train Class {c}")
    plt.plot(curr_epochs, d_tra_all, color='black', linestyle='-', linewidth=3, marker=marker_opt, label="Train Mean (All)")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.title(f"Training Dice Only - {title_suffix}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f"dice_train_only_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 5. Validation Dice ---
    plt.figure(figsize=(10, 6))
    for c in range(1, num_classes):
        cc = color_palette(c)
        plt.plot(curr_epochs, d_val[:, c], color=cc, linestyle='--', marker=val_marker_opt, label=f"Val Class {c}")
    plt.plot(curr_epochs, d_val_all, color='black', linestyle='--', linewidth=3, marker=val_marker_opt, label="Val Mean (All)")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.title(f"Validation Dice Only - {title_suffix}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f"dice_val_only_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()


# --- Overall Plots ---
plot_metrics(
    epochs,
    mean_loss_tra, mean_loss_val,
    class_dice_tra, class_dice_val,
    mean_dice_tra_all, mean_dice_val_all,
    file_suffix="overall",
    title_suffix=f"Overall ({total_epochs} Epochs)",
    is_window=False
)

# --- 10-Epoch window plots ---
chunk_size = 10
for start in range(0, total_epochs, chunk_size):
    end = min(start + chunk_size, total_epochs)
    subset_epochs = list(range(start + 1, end + 1))
    
    plot_metrics(
        subset_epochs,
        mean_loss_tra[start:end], mean_loss_val[start:end],
        class_dice_tra[start:end], class_dice_val[start:end],
        mean_dice_tra_all[start:end], mean_dice_val_all[start:end],
        file_suffix=f"epochs_{start+1:03d}_to_{end:03d}",
        title_suffix=f"Epochs {start+1} to {end}",
        is_window=True
    )

print(f"Successfully generated all 5 distinct plot categories in {results_dir}/")