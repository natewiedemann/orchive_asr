#!/bin/bash
#SBATCH --job-name=whisp_multi
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH -o /home/us80abag/logs/%j.out
#SBATCH -e /home/us80abag/logs/%j.err

set -e

INPUT_DIR="$1" 
DIR_NAME=$(basename "$INPUT_DIR")
HOME_OUT="/home/us80abag/whisper_output/$DIR_NAME"
mkdir -p "$HOME_OUT"

LOCAL_SCRATCH="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$LOCAL_SCRATCH/output"

# Background rsync loop (Every 30 mins)
(
  while [ -d "$LOCAL_SCRATCH" ]; do
    rsync -au "$LOCAL_SCRATCH/output/" "$HOME_OUT/"
    sleep 1800
  done
) &
SYNC_PID=$!

source /home/us80abag/venv/bin/activate

# CUDA path injection
export NV_LIBS=$(python3 -c "import nvidia.cublas.lib as lib; import os; print(os.path.dirname(lib.__file__))")
export NV_DNN=$(python3 -c "import nvidia.cudnn.lib as lib; import os; print(os.path.dirname(lib.__file__))")
export LD_LIBRARY_PATH=$NV_LIBS:$NV_DNN:$LD_LIBRARY_PATH

cd /home/us80abag/orcas

echo "===== START TIME ====="
date
echo "===== SLURM JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "===== GPU INFO ====="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "===================="

# Background GPU logger
nvidia-smi dmon -s pucvmet -d 5 -o DT > "/home/us80abag/logs/${SLURM_JOB_ID}_gpu_dmon.log" &
DMON_PID=$!

echo "Starting Multi-Worker Batch: $DIR_NAME"

# Using 2 GPUs, 2 workers per GPU (Total 4)
python -u whisper_batch.py "$INPUT_DIR" \
    --out_dir "$LOCAL_SCRATCH/output" \
    --scratch_dir "$LOCAL_SCRATCH" \
    --model large-v3 \
    --gpus 0,1 \
    --workers_per_gpu 2

# Cleanup
kill $SYNC_PID || true
kill $DMON_PID || true

echo "Finalizing data sync..."
rsync -au "$LOCAL_SCRATCH/output/" "$HOME_OUT/"
rm -rf "$LOCAL_SCRATCH"

# Remove local temp files from previous sync
TMP_META_DIR="${HOME_OUT}/tmp_meta"

if [[ -d "$TMP_META_DIR" && "$TMP_META_DIR" == "$HOME_OUT"/tmp_meta ]]; then
    rm -rf -- "$TMP_META_DIR"
fi

echo "done with processing jobID: $SLURM_JOB_ID"
date