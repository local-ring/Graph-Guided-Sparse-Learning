#!/bin/bash

#SBATCH -J gfl_exp
#SBATCH -p general
#SBATCH -o filename_%j.out      # stdout
#SBATCH -e filename_%j.err      # stderr
#SBATCH --mail-type=END,FAIL     # or ALL
#SBATCH --mail-user=aolli@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64      # threads per task
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH -A r00939   # your allocation

# 1) Load Miniconda module
module purge
module load miniconda

# 2) Initialize `conda` in this shell (so `conda activate` works)
eval "$(conda shell.bash hook)"

# 3) Activate your pre-built env
conda activate gfl

python setup.py build_ext --inplace
# 4) Move to your code directory (where you ran sbatch)
cd "$SLURM_SUBMIT_DIR"

# Make sure Python can find sparse_module.so
export PYTHONPATH="$PWD:$PYTHONPATH"

# 5) Run your Python script under srun
#    --cpus-per-task ensures Python sees 16 cores if using multithreading
srun python run_experiment.py
