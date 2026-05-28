#!/bin/bash
# ============================================================
#  SLURM Job Script — LSTM Training
#  Project : IDS-KMUTT Hybrid Intrusion Detection System
#  Author  : Darren Touopi
#  Node    : gpu4090 (NVIDIA RTX 4090, 24GB VRAM)
#  Submit  : sbatch lstm_job.sbatch
# ============================================================

#SBATCH --job-name=ids_lstm_train
#SBATCH --partition=gpu4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/storage/slurm/home/darrentouopi@cpe.kmutt.ac.th/ids_lstm_%j.out
#SBATCH --error=/storage/slurm/home/darrentouopi@cpe.kmutt.ac.th/ids_lstm_%j.err

echo "============================================"
echo "  IDS-KMUTT — LSTM Training Job"
echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Node         : $SLURMD_NODENAME"
echo "CPUs alloc   : $SLURM_CPUS_PER_TASK"
echo "Memory alloc : 32G"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time   : $(date)"
echo "============================================"

# Activate virtual environment
source ~/venv/bin/activate

# Install TensorFlow if not already installed
pip install tensorflow --quiet

# Verify setup
echo ""
echo "Python     : $(which python)"
echo "Python ver : $(python --version)"
echo "TF version : $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null)"
echo "GPU check  : $(python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))' 2>/dev/null)"
echo ""

cd ~/

python train_lstm.py

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Job finished at: $(date)"
echo "  Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
