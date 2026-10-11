#!/usr/bin/env bash
# The clean baseline with only one change at a time, to see what every change does on its own:
#   windowing      3 HU windows as input channels (SEGTHOR_MULTI, see hu_window/)
#   weight_decay   weight decay (L2) of 0.0001, with a learning rate of 0.00005 instead of 0.0005
#   augmentation   spatial augmentation (small rotations, scaling and shifts)
#   weighted_loss  Dice + class-weighted cross-entropy
#   2_5d           2.5D: the slices above and below as 2 extra input channels
#
# Everything else is the same as the clean baseline (results/baseline_clean_100ep_cpu):
# 100 epochs, full 256x256 slices (no crop), empty slices removed from the training set,
# cross-entropy, Adam with learning rate 0.0005, batch size 8, and the same 10 validation patients.

set -e

SOURCE=data/segthor_train_full
RESULTS=results/single_changes
EPOCHS=100
RUNS="windowing weight_decay augmentation weighted_loss 2_5d"

# 1. Slice the data, with the same split as SEGTHOR_CLEAN (10 validation patients, fold 0)
if [ ! -d data/SEGTHOR_CLEAN ]; then
    rm -rf data/SEGTHOR_CLEAN_tmp
    python slice_segthor.py --source_dir $SOURCE --dest_dir data/SEGTHOR_CLEAN_tmp \
        --shape 256 256 --retains 10 -p 6
    mv data/SEGTHOR_CLEAN_tmp data/SEGTHOR_CLEAN
fi

if [ ! -d data/SEGTHOR_MULTI ]; then
    rm -rf data/SEGTHOR_MULTI_tmp
    python hu_window/slice_segthor_window.py --source_dir $SOURCE --dest_dir data/SEGTHOR_MULTI_tmp \
        --shape 256 256 --retains 10 -p 6 --windows -150 250 -1000 -200 100 500
    mv data/SEGTHOR_MULTI_tmp data/SEGTHOR_MULTI
fi

# 2. Train one model per change
for run in $RUNS; do
    case $run in
        windowing) options="--dataset SEGTHOR_MULTI" ;;
        weight_decay) options="--dataset SEGTHOR_CLEAN --lr 0.00005 --weight_decay 0.0001" ;;
        augmentation) options="--dataset SEGTHOR_CLEAN --augment" ;;
        weighted_loss) options="--dataset SEGTHOR_CLEAN --weighted_dice_loss" ;;
        2_5d) options="--dataset SEGTHOR_CLEAN --is_25d" ;;
    esac

    if [ ! -f $RESULTS/$run/finished ]; then
        python -O main.py $options --epochs $EPOCHS --mode full --crop_size 0 --filter_empty \
            --dest $RESULTS/$run --gpu
        touch $RESULTS/$run/finished
    fi
done
