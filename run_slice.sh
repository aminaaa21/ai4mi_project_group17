#!/bin/bash
#SBATCH --job-name=segthor_slice_robust
#SBATCH --partition=rome
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

# 4. Fast, quiet package check inside the 3.11 environment
pip install torch --index-url https://download.pytorch.org/whl/cpu --no-deps --quiet
pip install scikit-image tqdm nibabel numpy --quiet

# 5. Clean up old output folders
rm -rf data/SEGTHOR_tmp data/SEGTHOR

# 6. Execute slicing script using active environment's Python
python slice_segthor.py --source_dir data/segthor_part1 --dest_dir data/SEGTHOR_tmp --shape 256 256 --retain 10

# 7. Validate output and complete README workflow
if [ -d "data/SEGTHOR_tmp" ]; then
    mv data/SEGTHOR_tmp data/SEGTHOR
    echo "=== Slicing Completed Successfully ==="
else
    echo "ERROR: data/SEGTHOR_tmp was not created!" >&2
    exit 1
fi

echo "=== Job finished at $(date) ==="