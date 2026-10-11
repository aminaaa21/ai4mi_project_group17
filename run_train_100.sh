#!/bin/bash
#SBATCH --job-name=segthor_cpu_100
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=train_cpu_%j.log
#SBATCH --error=train_cpu_%j.err

# Exit on error
set -euo pipefail

echo "=== Training Job Started on Genoa CPU at $(date) ==="

# 1. Load environment
module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

# 2. Run training with CPU configuration; -dest results/baseline_clean_100ep_cpu \
python main.py \
    --dataset SEGTHOR_CLEAN \
    --epochs 100 \
    --mode full \
    --dest results//baseline_clean_100ep_cpu \
    --crop_size 0 \
    --filter_empty


echo "=== Training Job Finished at $(date) ==="