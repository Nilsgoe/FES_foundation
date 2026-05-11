#!/bin/bash -l
#SBATCH --job-name=triala_metad
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

mkdir -p outputs logs structures

source "${VENV_DIR}/bin/activate"

run_name="${RUN_NAME_OVERRIDE:-${RUN_NAME}}"
start_file="${START_FILE_OVERRIDE:-${START_FILE}}"
mode="${MODE_OVERRIDE:-${MODE}}"
steps="${STEPS_PER_CHUNK:-1000000}"
chunk="${CHUNK_ID:-0}"

echo "Run ${run_name} mode ${mode} chunk ${chunk} on ${SLURMD_NODENAME}" >&2
echo "Start file: ${start_file}" >&2
echo "Steps per chunk: ${steps}" >&2
rocm-smi > "logs/gpu_util_${run_name}_chunk${chunk}.txt" 2>&1 || true

python run_trialanine.py \
    --mode "${mode}" \
    --model-kind "${MODEL_KIND}" \
    --start-file "${start_file}" \
    --run-name "${run_name}" \
    --chunk-id "${chunk}" \
    --steps-per-chunk "${steps}"
