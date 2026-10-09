#!/usr/bin/env bash
# The full model (run_full_model.sh) with one change left out at a time, to see how much every
# change adds when all the others are already there:
#   no_windowing      SEGTHOR_CLEAN (min-max normalization) instead of the 3 HU windows
#   no_weight_decay   no weight decay, and the normal learning rate of 0.0005 instead of 0.00005
#   no_augmentation   no spatial augmentation
#   no_weighted_loss  cross-entropy instead of Dice + class-weighted cross-entropy
#
# Everything else is the same as the full model: 100 epochs, full 256x256 slices (no crop),
# empty slices removed from the training set, batch size 8, and the same 10 validation patients.

set -e

RESULTS=results/leave_one_out
EPOCHS=100
RUNS="no_windowing no_weight_decay no_augmentation no_weighted_loss"

for run in $RUNS; do
    case $run in
        no_windowing) options="--dataset SEGTHOR_CLEAN --lr 0.00005 --weight_decay 0.0001 --augment --weighted_dice_loss" ;;
        no_weight_decay) options="--dataset SEGTHOR_MULTI --augment --weighted_dice_loss" ;;
        no_augmentation) options="--dataset SEGTHOR_MULTI --lr 0.00005 --weight_decay 0.0001 --weighted_dice_loss" ;;
        no_weighted_loss) options="--dataset SEGTHOR_MULTI --lr 0.00005 --weight_decay 0.0001 --augment" ;;
    esac

    if [ ! -f $RESULTS/$run/finished ]; then
        python -O main.py $options --epochs $EPOCHS --mode full --crop_size 0 --filter_empty \
            --dest $RESULTS/$run --gpu
        touch $RESULTS/$run/finished
    fi
done
