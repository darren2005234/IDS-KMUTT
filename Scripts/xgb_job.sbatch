#!/bin/bash
# ============================================================
#  SLURM Job Script — XGBoost + Optuna Training
#  Project : IDS-KMUTT Hybrid Intrusion Detection System
#  Author  : Darren Touopi
#  Node    : cpu partition (max 16 CPUs, max 24h)
#  Submit  : sbatch xgb_job.sbatch
# ============================================================

#SBATCH --job-name=ids_xgb_train
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "============================================"
echo "  IDS-KMUTT — XGBoost + Optuna Training"
echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Node         : $SLURMD_NODENAME"
echo "CPUs alloc   : $SLURM_CPUS_PER_TASK"
echo "Start time   : $(date)"
echo "============================================"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ids_env

# Install Optuna if not already in the environment
pip install optuna xgboost --quiet

echo ""
echo "Python : $(which python)"
echo "XGBoost: $(python -c 'import xgboost; print(xgboost.__version__)')"
echo "Optuna : $(python -c 'import optuna; print(optuna.__version__)')"
echo ""

cd ~/ids_project/

python train_xgb.py

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Job finished at: $(date)"
echo "  Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
