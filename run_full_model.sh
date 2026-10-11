#!/usr/bin/env bash
# The full model: the clean baseline with all the changes of run_single_changes.sh at once:
#   windowing      3 HU windows as input channels (SEGTHOR_MULTI, see hu_window/)
#   weight_decay   weight decay (L2) of 0.0001, with a learning rate of 0.00005 instead of 0.0005
#   augmentation   spatial augmentation (small rotations, scaling and shifts)
#   weighted_loss  Dice + class-weighted cross-entropy
#
# Everything else is the same as the clean baseline (results/baseline_clean_100ep_cpu):
# 100 epochs, full 256x256 slices (no crop), empty slices removed from the training set,
# batch size 8, and the same 10 validation patients.
# dice_val.npy is saved after every epoch, so the first 25 epochs can also be compared.

set -e

SOURCE=data/segthor_train_full
RESULTS=results/full_model
EPOCHS=100

# 1. Slice the data, with the same split as SEGTHOR_CLEAN (10 validation patients, fold 0)
if [ ! -d data/SEGTHOR_MULTI ]; then
    rm -rf data/SEGTHOR_MULTI_tmp
    python hu_window/slice_segthor_window.py --source_dir $SOURCE --dest_dir data/SEGTHOR_MULTI_tmp \
        --shape 256 256 --retains 10 -p 6 --windows -150 250 -1000 -200 100 500
    mv data/SEGTHOR_MULTI_tmp data/SEGTHOR_MULTI
fi

# 2. Train
if [ ! -f $RESULTS/finished ]; then
    python -O main.py --dataset SEGTHOR_MULTI --lr 0.00005 --weight_decay 0.0001 --augment --weighted_dice_loss \
        --epochs $EPOCHS --mode full --crop_size 0 --filter_empty --dest $RESULTS --gpu
    touch $RESULTS/finished
fi
