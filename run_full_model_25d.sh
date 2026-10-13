#!/usr/bin/env bash
# The full model with 2.5D, and the same model with one change left out at a time.
# 2.5D does not work with the 3 HU windows (3 slices x 3 windows = 9 input channels, too many for
# ENet), so the windowing here is only the soft tissue window -150..250 HU (SEGTHOR_WINDOW).
#   full              all the changes below
#   no_windowing      SEGTHOR_CLEAN (min-max normalization) instead of the soft tissue window
#   no_weight_decay   no weight decay, and the normal learning rate of 0.0005 instead of 0.00005
#                     (so 2 changes left out at once, see the 2 runs below)
#   no_weight_decay_lr_kept  no weight decay, the learning rate stays 0.00005
#   no_lower_lr       the normal learning rate of 0.0005, weight decay stays
#   no_augmentation   no spatial augmentation
#   no_weighted_loss  cross-entropy instead of Dice + class-weighted cross-entropy
#   no_2_5d           only the current slice as input (1 channel)
#
# Everything else is the same as the other runs: 100 epochs, full 256x256 slices (no crop),
# empty slices removed from the training set, batch size 8, and the same 10 validation patients.

set -e

SOURCE=data/segthor_train_full
RESULTS=results/full_model_25d
EPOCHS=100
RUNS="full no_windowing no_weight_decay no_augmentation no_weighted_loss no_2_5d no_weight_decay_lr_kept no_lower_lr"

# 1. Slice the data, with the same split as SEGTHOR_CLEAN (10 validation patients, fold 0)
if [ ! -d data/SEGTHOR_WINDOW ]; then
    rm -rf data/SEGTHOR_WINDOW_tmp
    python hu_window/slice_segthor_window.py --source_dir $SOURCE --dest_dir data/SEGTHOR_WINDOW_tmp \
        --shape 256 256 --retains 10 -p 6 --windows -150 250
    mv data/SEGTHOR_WINDOW_tmp data/SEGTHOR_WINDOW
fi

# 2. Train
for run in $RUNS; do
    case $run in
        full) options="--dataset SEGTHOR_WINDOW --is_25d --lr 0.00005 --weight_decay 0.0001 --augment --weighted_dice_loss" ;;
        no_windowing) options="--dataset SEGTHOR_CLEAN --is_25d --lr 0.00005 --weight_decay 0.0001 --augment --weighted_dice_loss" ;;
        no_weight_decay) options="--dataset SEGTHOR_WINDOW --is_25d --augment --weighted_dice_loss" ;;
        no_augmentation) options="--dataset SEGTHOR_WINDOW --is_25d --lr 0.00005 --weight_decay 0.0001 --weighted_dice_loss" ;;
        no_weighted_loss) options="--dataset SEGTHOR_WINDOW --is_25d --lr 0.00005 --weight_decay 0.0001 --augment" ;;
        no_2_5d) options="--dataset SEGTHOR_WINDOW --lr 0.00005 --weight_decay 0.0001 --augment --weighted_dice_loss" ;;
        no_weight_decay_lr_kept) options="--dataset SEGTHOR_WINDOW --is_25d --lr 0.00005 --augment --weighted_dice_loss" ;;
        no_lower_lr) options="--dataset SEGTHOR_WINDOW --is_25d --weight_decay 0.0001 --augment --weighted_dice_loss" ;;
    esac

    if [ ! -f $RESULTS/$run/finished ]; then
        python -O main.py $options --epochs $EPOCHS --mode full --crop_size 0 --filter_empty \
            --dest $RESULTS/$run --gpu
        touch $RESULTS/$run/finished
    fi
done
