#!/usr/bin/env bash
# Turn the predictions of the hu_window runs into 3D volumes (.nii.gz) and evaluate them with
# evaluate_3d.py (3D Dice, HD95, MSD).
#
# Every patient is in the validation set of exactly one fold, so the 4 folds together give one
# volume for each of the 40 patients, per method.

set -e

RESULTS=results/hu_window_full
VOLUMES=volumes/hu_window
GT_DIR=data/segthor_train_full/train
FOLDS="0 1 2 3"
METHODS="baseline window wide multi multi_dice"

for method in $METHODS; do
    # 1. Stitch the 2D predictions of the best epoch back to 3D
    if [ ! -d $VOLUMES/$method ]; then
        for fold in $FOLDS; do
            python stitch.py --data_folder $RESULTS/${method}_fold$fold/best_epoch/val \
                --dest_folder $VOLUMES/${method}_tmp \
                --num_classes 255 --grp_regex "(Patient_\d\d)_\d\d\d\d" \
                --source_scan_pattern "$GT_DIR/{id_}/GT.nii.gz"
        done
        mv $VOLUMES/${method}_tmp $VOLUMES/$method
    fi

    # 2. 3D metrics per patient and per class
    if [ ! -d $RESULTS/metrics/$method ]; then
        python evaluate_3d.py --pred_dir $VOLUMES/$method --gt_dir $GT_DIR \
            --output_dir $RESULTS/metrics/${method}_tmp
        mv $RESULTS/metrics/${method}_tmp $RESULTS/metrics/$method
    fi
done
