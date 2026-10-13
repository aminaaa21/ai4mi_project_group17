#!/usr/bin/env bash
# Checks on the final model (3 HU windows, augmentation, Dice + weighted cross-entropy, learning rate
# 0.0005, no weight decay), to make sure it is the best combination:
#   no_weighted_loss  plain cross-entropy instead of Dice + weighted cross-entropy (in the 3-window
#                     ablation the full model got better without it)
#
# Everything else is the same as the final model: 100 epochs, full 256x256 slices (no crop),
# empty slices removed from the training set, batch size 8, and the same 10 validation patients.

set -e

RESULTS=results/final_check
EPOCHS=100
RUNS="no_weighted_loss"

for run in $RUNS; do
    case $run in
        no_weighted_loss) options="--dataset SEGTHOR_MULTI --augment" ;;
    esac

    if [ ! -f $RESULTS/$run/finished ]; then
        python -O main.py $options --epochs $EPOCHS --mode full --crop_size 0 --filter_empty \
            --dest $RESULTS/$run --gpu
        touch $RESULTS/$run/finished
    fi
done
