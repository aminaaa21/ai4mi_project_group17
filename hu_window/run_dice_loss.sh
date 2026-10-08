#!/usr/bin/env bash
# Follow-up of run_all.sh: retrain the "3 windows" method with cross-entropy + soft Dice loss
# (CrossEntropyDice in losses.py) instead of cross-entropy only, on the same 4 folds and data.
# Then compare: baseline (CE), 3 windows (CE), 3 windows (CE + Dice).

set -e

DATA=data/SEGTHOR_FULL
RESULTS=results/hu_window_full
EPOCHS=25
FOLDS="0 1 2 3"

# 1. Train (same data and settings as the "multi" runs, only the loss changes)
for fold in $FOLDS; do
    if [ ! -f $RESULTS/multi_dice_fold$fold/finished ]; then
        python -O main.py --dataset SEGTHOR_MULTI --data_dir ${DATA}_multi_fold$fold \
            --loss ce_dice --epochs $EPOCHS --dest $RESULTS/multi_dice_fold$fold --gpu --crop_size 0 --no-filter_empty
        touch $RESULTS/multi_dice_fold$fold/finished
    fi
done

# 2. 3D Dice per patient, grouped by contrast status (baseline first: it is the reference)
GT_DIRS=""
for fold in $FOLDS; do
    GT_DIRS="$GT_DIRS ${DATA}_baseline_fold$fold/val/gt"
done

METHOD_ARGS=""
for method in baseline multi multi_dice; do
    METHOD_ARGS="$METHOD_ARGS --method $method"
    for fold in $FOLDS; do
        METHOD_ARGS="$METHOD_ARGS $RESULTS/${method}_fold$fold/best_epoch/val"
    done
done

python hu_window/evaluate_contrast.py --contrast_csv $RESULTS/contrast.csv \
    --gt_dirs $GT_DIRS $METHOD_ARGS --dest $RESULTS/dice_loss/dice_3d.csv

# 3. Charts
python hu_window/plot_results.py --results_dir $RESULTS --data_prefix $DATA \
    --dice_csv $RESULTS/dice_loss/dice_3d.csv --dest_dir $RESULTS/dice_loss/figures
