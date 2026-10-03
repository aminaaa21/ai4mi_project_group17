#!/bin/bash
#SBATCH --job-name=segthor_slice_full
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=slice_job_%j.log
#SBATCH --error=slice_job_%j.err

# Exit on error, unset variables, or pipe failures
set -euo pipefail

echo "=== Job Started at $(date) ==="

# 1. Clear environment and explicitly load Python 3.11 module
module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0

# 2. Activate virtual environment
source venv/bin/activate

# 3. Print active Python version to verify module loaded correctly
python --version

# 4. Clean up old output folders
rm -rf data/SEGTHOR_FULL_tmp data/SEGTHOR_FULL

# 5. Execute slicing script using the full dataset path
python slice_segthor.py --source_dir data/segthor_train_full --dest_dir data/SEGTHOR_FULL_tmp --shape 256 256 --retains 10

# 6. Validate output and move to final destination
if [ -d "data/SEGTHOR_FULL_tmp" ]; then
    mv data/SEGTHOR_FULL_tmp data/SEGTHOR_FULL
    echo "=== Slicing Completed Successfully ==="
else
    echo "ERROR: data/SEGTHOR_FULL_tmp was not created!" >&2
    exit 1
fi

echo "=== Job finished at $(date) ==="