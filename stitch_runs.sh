#!/usr/bin/env bash
# Stitch the best-epoch predictions of the runs of run_single_changes.sh, run_full_model.sh,
# run_leave_one_out.sh and run_full_model_25d.sh back to 3D volumes (.nii.gz), for the 3D
# evaluation with evaluate_3d.py.
# The volumes of results/<name> go to volumes/<name>. Runs that did not finish are left out.

set -e

RUNS="single_changes/windowing single_changes/weight_decay single_changes/augmentation
      single_changes/weighted_loss single_changes/2_5d single_changes/windowing_1ch full_model
      leave_one_out/no_windowing leave_one_out/no_weight_decay leave_one_out/no_augmentation
      leave_one_out/no_weighted_loss
      full_model_25d/full full_model_25d/no_windowing full_model_25d/no_weight_decay
      full_model_25d/no_augmentation full_model_25d/no_weighted_loss full_model_25d/no_2_5d
      single_changes/weight_decay_only single_changes/lower_lr
      full_model_25d/no_weight_decay_lr_kept full_model_25d/no_lower_lr
      leave_one_out/no_lower_lr leave_one_out/no_weight_decay_lr_kept"

for run in $RUNS; do
    if [ -f results/$run/finished ] && [ ! -d volumes/$run ]; then
        rm -rf volumes/${run}_tmp
        python stitch.py --data_folder results/$run/best_epoch/val \
            --dest_folder volumes/${run}_tmp \
            --num_classes 255 --grp_regex "(Patient_\d\d)_\d\d\d\d" \
            --source_scan_pattern "data/segthor_train_full/train/{id_}/GT.nii.gz"
        mv volumes/${run}_tmp volumes/$run
    fi
done
