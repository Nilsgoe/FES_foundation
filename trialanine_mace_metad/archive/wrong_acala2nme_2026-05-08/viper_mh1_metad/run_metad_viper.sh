#!/bin/bash -l
#SBATCH --job-name=triala_mh1_viper_metad
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
export HIPRTC_COMPILE_OPTIONS_APPEND="--no-default-config"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="/ptmp/ngoen/biase_venv_sol3r/lib/python3.12/site-packages:/u/ngoen/software/biASE${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p logs outputs structures

source "${VENV_DIR}/bin/activate"

chunk_id="${CHUNK_ID:?CHUNK_ID must be set}"
steps_per_chunk="${STEPS_PER_CHUNK_OVERRIDE:-${STEPS_PER_CHUNK}}"

echo "Run ${RUN_NAME} chunk ${chunk_id} on ${SLURMD_NODENAME}" >&2
echo "Steps per chunk: ${steps_per_chunk}" >&2
rocm-smi > "logs/gpu_util_${RUN_NAME}_chunk${chunk_id}_${SLURM_JOB_ID}.txt" 2>&1 || true

python run_mh1_metad_viper.py \
  --start-file "${START_FILE}" \
  --run-name "${RUN_NAME}" \
  --chunk-id "${chunk_id}" \
  --steps-per-chunk "${steps_per_chunk}" \
  --temperature "${TEMPERATURE_K}" \
  --timestep-fs "${TIMESTEP_FS}" \
  --device cuda
