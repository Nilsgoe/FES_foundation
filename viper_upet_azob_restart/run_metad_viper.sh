#!/bin/bash -l
#SBATCH --job-name=metad_azob
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
if [[ "${MODEL_KIND}" == "upet" ]]; then
    module load gcc/14 rocm/6.4 python-waterboa/2024.06
    export HIPRTC_COMPILE_OPTIONS_APPEND="--no-default-config"
elif [[ "${MODEL_KIND}" == "sol3r" ]]; then
    module load gcc/14 rocm/7.2 python-waterboa/2024.06
    export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_HOME}/lib/llvm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LLVM_PATH=/mpcdf/soft/RHEL_9/packages/x86_64/rocm/7.2.1/llvm
    export HIPRTC_COMPILE_OPTIONS_APPEND="--no-default-config"
else
    echo "Unknown MODEL_KIND=${MODEL_KIND}" >&2
    exit 2
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="/u/ngoen/software/biASE${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p outputs logs

source "${VENV_DIR}/bin/activate"

echo "Run ${RUN_NAME} chunk ${CHUNK_ID} on ${SLURMD_NODENAME}" >&2
echo "Steps per chunk: ${STEPS_PER_CHUNK}" >&2
echo "Restart offset: ${RESTART_OFFSET:--1}" >&2
rocm-smi > "logs/gpu_util_${RUN_NAME}_chunk${CHUNK_ID}.txt" 2>&1 || true

python run_metad.py \
    --model-kind "${MODEL_KIND}" \
    --system "${SYSTEM_NAME}" \
    --cv-mode "${CV_MODE}" \
    --start-file "${START_FILE}" \
    --chunk-id "${CHUNK_ID}" \
    --steps-per-chunk "${STEPS_PER_CHUNK}" \
    --restart-offset "${RESTART_OFFSET:--1}"
