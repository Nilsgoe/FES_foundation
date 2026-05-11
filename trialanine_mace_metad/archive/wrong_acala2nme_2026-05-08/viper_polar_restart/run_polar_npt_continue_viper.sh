#!/bin/bash -l
#SBATCH --job-name=triala_polar_viper_npt
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

run_name="${RUN_NAME_OVERRIDE:-${RUN_NAME}}"
job_name="${JOB_NAME_OVERRIDE:-${JOB_NAME}}"
model_path="${MODEL_PATH_OVERRIDE:-${MODEL_PATH}}"
input_traj="${INPUT_TRAJ_OVERRIDE:-${INPUT_TRAJ}}"
fallback_start="${FALLBACK_START_OVERRIDE:-${FALLBACK_START}}"
npt_output="${NPT_OUTPUT_OVERRIDE:-${NPT_OUTPUT}}"
temperature_k="${TEMPERATURE_K_OVERRIDE:-${TEMPERATURE_K}}"
pressure_bar="${PRESSURE_BAR_OVERRIDE:-${PRESSURE_BAR}}"
timestep_fs="${TIMESTEP_FS_OVERRIDE:-${TIMESTEP_FS}}"
npt_steps="${NPT_STEPS_OVERRIDE:-${NPT_STEPS}}"
default_dtype="${DEFAULT_DTYPE_OVERRIDE:-${DEFAULT_DTYPE}}"

module purge
module load gcc/14 rocm/6.4 python-waterboa/2024.06
export HIPRTC_COMPILE_OPTIONS_APPEND="--no-default-config"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs outputs structures models

source "${VENV_DIR}/bin/activate"

echo "Running ${run_name} on ${SLURMD_NODENAME}" >&2
echo "Project: ${PROJECT_DIR}" >&2
echo "Model: ${model_path}" >&2
echo "Input traj: ${input_traj}" >&2
echo "Fallback start: ${fallback_start}" >&2
echo "Remaining NPT steps: ${npt_steps}" >&2
rocm-smi > "logs/gpu_util_${run_name}_${SLURM_JOB_ID}.txt" 2>&1 || true

python run_polar_npt_continue.py \
  --input-traj "${input_traj}" \
  --fallback-start "${fallback_start}" \
  --model-path "${model_path}" \
  --output-prefix "outputs/${run_name}_${SLURM_JOB_ID}" \
  --npt-output "${npt_output}" \
  --npt-steps "${npt_steps}" \
  --temperature "${temperature_k}" \
  --pressure-bar "${pressure_bar}" \
  --timestep-fs "${timestep_fs}" \
  --default-dtype "${default_dtype}"
