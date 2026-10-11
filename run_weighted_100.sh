#!/bin/bash
#SBATCH --job-name=weighted_loss_100
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=weighted_100_%j.log
#SBATCH --error=weighted_100_%j.err

set -euo pipefail

module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0
source ai4mi/bin/activate

python main.py \
    --dataset SEGTHOR_CLEAN \
    --epochs 100 \
    --mode full \
    --dest results/weighted_dice_clean_100ep \
    --crop_size 0 \
    --filter_empty \
    --weighted_dice_loss
