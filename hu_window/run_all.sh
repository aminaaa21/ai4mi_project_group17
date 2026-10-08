#!/usr/bin/env bash
# Full experiment on the 40 patients of segthor_train_full, 4 ways to normalize the CT:
#   baseline  min-max normalization of every scan (original slice_segthor.py)
#   window    soft tissue window -150..250 HU
#   wide      wider window -150..400 HU (does not clip the contrast-enhanced aorta)
#   multi     3 windows at once, one per input channel:
#               soft tissue -150..250 HU, air/lungs -1000..-200 HU, contrast/bone 100..500 HU
# evaluated separately on contrast-enhanced and non-enhanced patients.
#
# We use a 4-fold cross-validation (10 validation patients per fold): every patient is in the
# validation set of exactly one fold, so all 40 patients can be evaluated.

set -e

SOURCE=data/segthor_train_full
DATA=data/SEGTHOR_FULL
RESULTS=results/hu_window_full
EPOCHS=25
FOLDS="0 1 2 3"
METHODS="baseline window wide multi"
RETAINS=10  # Number of validation patients per fold (40 patients / 4 folds)

mkdir -p $RESULTS

# 1. Which patients are contrast-enhanced?
if [ ! -f $RESULTS/contrast.csv ]; then
    python hu_window/detect_contrast.py --source_dir $SOURCE --dest $RESULTS/contrast.csv
fi

# 2. Slice every fold once per method
for fold in $FOLDS; do
    for method in $METHODS; do
        dest=${DATA}_${method}_fold$fold
        if [ -d $dest ]; then
            continue
        fi

        rm -rf ${dest}_tmp
        case $method in
            baseline)
                python slice_segthor.py --source_dir $SOURCE --dest_dir ${dest}_tmp \
                    --shape 256 256 --retains $RETAINS --fold $fold -p 6 ;;
            window)
                python hu_window/slice_segthor_window.py --source_dir $SOURCE --dest_dir ${dest}_tmp \
                    --shape 256 256 --retains $RETAINS --fold $fold -p 6 --windows -150 250 ;;
            wide)
                python hu_window/slice_segthor_window.py --source_dir $SOURCE --dest_dir ${dest}_tmp \
                    --shape 256 256 --retains $RETAINS --fold $fold -p 6 --windows -150 400 ;;
            multi)
                python hu_window/slice_segthor_window.py --source_dir $SOURCE --dest_dir ${dest}_tmp \
                    --shape 256 256 --retains $RETAINS --fold $fold -p 6 \
                    --windows -150 250 -1000 -200 100 500 ;;
        esac
        mv ${dest}_tmp $dest
    done
done

# 3. Train one model per fold and per method (same training code for all, only the data differs).
#    No crop and no empty slice filtering (main.py does both by default), as in our first runs.
#    One training at a time: running two at once on the GTX 1080 was not faster (the GPU is
#    already fully busy with one) and used a lot more RAM.
for fold in $FOLDS; do
    for method in $METHODS; do
        case $method in
            baseline) dataset=SEGTHOR ;;
            window|wide) dataset=SEGTHOR_WINDOW ;;
            multi) dataset=SEGTHOR_MULTI ;;  # 3 input channels
        esac

        if [ ! -f $RESULTS/${method}_fold$fold/finished ]; then
            python -O main.py --dataset $dataset --data_dir ${DATA}_${method}_fold$fold \
                --epochs $EPOCHS --dest $RESULTS/${method}_fold$fold --gpu --crop_size 0 --no-filter_empty
            touch $RESULTS/${method}_fold$fold/finished
        fi
    done
done

# 4. 3D Dice per patient, grouped by contrast status (baseline first: it is the reference)
GT_DIRS=""
for fold in $FOLDS; do
    GT_DIRS="$GT_DIRS ${DATA}_baseline_fold$fold/val/gt"
done

METHOD_ARGS=""
for method in $METHODS; do
    METHOD_ARGS="$METHOD_ARGS --method $method"
    for fold in $FOLDS; do
        METHOD_ARGS="$METHOD_ARGS $RESULTS/${method}_fold$fold/best_epoch/val"
    done
done

python hu_window/evaluate_contrast.py --contrast_csv $RESULTS/contrast.csv \
    --gt_dirs $GT_DIRS $METHOD_ARGS --dest $RESULTS/dice_3d.csv

# 5. Charts
python hu_window/plot_results.py --results_dir $RESULTS --data_prefix $DATA
