#!/bin/bash -l
#SBATCH --job-name=opt_sol3r_azob
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64000
#SBATCH --time=00:10:00
#SBATCH --partition=apu
#SBATCH --gres=gpu:1
#SBATCH -D ./
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

source ./config.env

module purge
module load gcc/14 rocm/7.2 python-waterboa/2024.06
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_HOME}/lib/llvm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LLVM_PATH=/mpcdf/soft/RHEL_9/packages/x86_64/rocm/7.2.1/llvm
export HIPRTC_COMPILE_OPTIONS_APPEND="--no-default-config"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="/u/ngoen/software/biASE${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p outputs logs
source "${VENV_DIR}/bin/activate"

echo "Optimizing azobenzene cis/trans with SO3LR on ${SLURMD_NODENAME}" >&2
rocm-smi > "logs/gpu_util_opt_${SLURM_JOB_ID}.txt" 2>&1 || true

python run_opt.py
