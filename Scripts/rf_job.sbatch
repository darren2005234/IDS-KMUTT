#!/bin/bash
# ============================================================
#  SLURM Job Script — Random Forest Training
#  Project : IDS-KMUTT Hybrid Intrusion Detection System
#  Author  : Darren Touopi
#  Node    : CPU01 (2× Intel Xeon E5-2698 v3 — 64 threads, 125 GB RAM)
#  Submit  : sbatch rf_job.sbatch
# ============================================================

#SBATCH --job-name=ids_rf_train
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ──────────────────────────────────────────────
# Environment info (written to job log)
# ──────────────────────────────────────────────
echo "============================================"
echo "  IDS-KMUTT — Random Forest Training Job"
echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Node         : $SLURMD_NODENAME"
echo "CPUs alloc   : $SLURM_CPUS_PER_TASK"
echo "Memory alloc : 60G"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "============================================"

# ──────────────────────────────────────────────
# Activate conda environment
# Option A — if conda is available via module:
#   module load anaconda
#   conda activate ids_env
#
# Option B — if you installed Miniconda in your home:
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate ids_env
#
# Uncomment the lines that match your cluster setup.
# If unsure, check with: module avail | grep -i conda
# ──────────────────────────────────────────────

# --- Option A (try this first) ---
# module load anaconda
# conda activate ids_env

# --- Option B (fallback) ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ids_env

# Verify Python
echo ""
echo "Python path  : $(which python)"
echo "Python ver   : $(python --version)"
echo ""

# ──────────────────────────────────────────────
# Run training script
# ──────────────────────────────────────────────
cd ~/ids_project/

python train_rf.py

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Job finished at: $(date)"
echo "  Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
