#!/usr/bin/env bash
# Ablation study with the normal learning rate of 0.0005 (the lower learning rate made every model worse).
# Full model: 3 HU windows + augmentation + Dice and weighted cross-entropy + weight decay 0.0001,
# learning rate 0.0005. That run already exists: results/leave_one_out/no_lower_lr. So is the full model
# without weight decay: results/leave_one_out/no_weight_decay (the final model).
# The other runs, with one change left out at a time:
#   no_weighted_loss  plain cross-entropy instead of Dice + weighted cross-entropy
#   no_augmentation   no spatial augmentation
#   no_windowing      SEGTHOR_CLEAN (min-max normalization) instead of the 3 HU windows
#
# Everything else is the same as the other runs: 100 epochs, full 256x256 slices (no crop),
# empty slices removed from the training set, batch size 8, and the same 10 validation patients.

set -e

RESULTS=results/ablation_lr0005
EPOCHS=100
RUNS="no_weighted_loss no_augmentation no_windowing"

for run in $RUNS; do
    case $run in
        no_weighted_loss) options="--dataset SEGTHOR_MULTI --weight_decay 0.0001 --augment" ;;
        no_augmentation) options="--dataset SEGTHOR_MULTI --weight_decay 0.0001 --weighted_dice_loss" ;;
        no_windowing) options="--dataset SEGTHOR_CLEAN --weight_decay 0.0001 --augment --weighted_dice_loss" ;;
    esac

    if [ ! -f $RESULTS/$run/finished ]; then
        python -O main.py $options --epochs $EPOCHS --mode full --crop_size 0 --filter_empty \
            --dest $RESULTS/$run --gpu
        touch $RESULTS/$run/finished
    fi
done
