#!/bin/bash -l
#SBATCH --job-name=metad_azob_2d_cont
#SBATCH --partition=gpubig
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=125000
#SBATCH --time=3-23:00:00
#SBATCH -o /nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene/logs/metad_azob_2d_cont_%x_%j.out
#SBATCH -e /nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene/logs/metad_azob_2d_cont_%x_%j.err

set -euo pipefail

module purge

project_dir=/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene
model_key=${MODEL_KEY:?MODEL_KEY must be set}
system_name=${SYSTEM_NAME:?SYSTEM_NAME must be set}
extra_steps=${EXTRA_STEPS:?EXTRA_STEPS must be set}

case "${model_key}" in
  off|omol|mh1)
    python_venv_dir=/fhi/home/ngoen/software/mace_cueq_venv
    ;;
  polar)
    python_venv_dir=/fhi/home/ngoen/software/biASE-venv
    ;;
  *)
    echo "Unsupported MODEL_KEY=${model_key}" >&2
    exit 1
    ;;
esac

mkdir -p "${project_dir}/outputs" "${project_dir}/logs"

scratch_dir=/scratch/ngoen_${SLURM_JOB_ID}_${model_key}_${system_name}_2d

export PYTHONPATH="/nexus/posix0/FHI-Theory/ngoen/biASE${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cleanup() {
  shopt -s nullglob
  for f in "${scratch_dir}/outputs"/*; do
    cp -f "${f}" "${project_dir}/outputs/" 2>/dev/null || true
  done
  for f in "${scratch_dir}/logs"/*; do
    cp -f "${f}" "${project_dir}/logs/" 2>/dev/null || true
  done
  shopt -u nullglob
  rm -rf "${scratch_dir}"
}
trap cleanup EXIT

mkdir -p "${scratch_dir}"
cd "${scratch_dir}"
mkdir -p outputs logs
rsync -a --exclude logs "${project_dir}/" .

source "${python_venv_dir}/bin/activate"

pattern="outputs/metad_azob_${system_name}_${model_key}_2d_*.traj"
shopt -s nullglob
matches=(${pattern})
shopt -u nullglob

if [ ${#matches[@]} -ne 1 ]; then
  echo "Expected exactly one match for ${pattern}, found ${#matches[@]}" >&2
  exit 1
fi

orig_traj="${matches[0]}"
stem=$(basename "${orig_traj}" .traj)
old_run_label="${stem#metad_azob_${system_name}_${model_key}_2d_}"

echo "Continuing ${model_key} ${system_name} 2d for ${extra_steps} steps" >&2
CUDA_VISIBLE_DEVICES=0 python run_metad.py \
  --system "${system_name}" \
  --model-key "${model_key}" \
  --cv-mode 2d \
  --run-label "${old_run_label}" \
  --steps "${extra_steps}" \
  --continue-run \
  --trajectory-loginterval 10

nvidia-smi > "logs/gpu_util_cont_2d_${model_key}_${system_name}.txt" 2>&1 || true
