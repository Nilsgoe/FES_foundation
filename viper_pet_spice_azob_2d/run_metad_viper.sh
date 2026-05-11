#!/bin/bash -l
#SBATCH --job-name=metad_azob_spice
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100000
#SBATCH --time=23:59:59
#SBATCH --partition=apu
#SBATCH --gres=gpu:1
#SBATCH -D ./
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

source ./config.env

module purge
module load gcc/14 rocm/6.4 python-waterboa/2024.06

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HIPRTC_COMPILE_OPTIONS_APPEND="--no-default-config"
export PYTHONPATH="/u/ngoen/software/biASE${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/ptmp/ngoen/.cache/huggingface_pet_spice
export HUGGINGFACE_HUB_CACHE=/ptmp/ngoen/.cache/huggingface_pet_spice/hub

mkdir -p outputs logs
source "${VENV_DIR}/bin/activate"

echo "Run ${RUN_NAME} chunk ${CHUNK_ID} on ${SLURMD_NODENAME}" >&2
echo "Steps per chunk: ${STEPS_PER_CHUNK}" >&2
rocm-smi > "logs/gpu_util_${RUN_NAME}_chunk${CHUNK_ID}.txt" 2>&1 || true

python run_metad.py \
  --system "${SYSTEM_NAME}" \
  --start-file "${START_FILE}" \
  --chunk-id "${CHUNK_ID}" \
  --steps-per-chunk "${STEPS_PER_CHUNK}"
