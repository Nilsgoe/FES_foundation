#!/bin/bash -l
#SBATCH --job-name=metad_azob
#SBATCH --partition=gpubig
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=125000
#SBATCH --time=3-23:00:00
#SBATCH --array=0-1%1
#SBATCH -o /nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene/logs/metad_azob_%x_%j.out
#SBATCH -e /nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene/logs/metad_azob_%x_%j.err

set -euo pipefail

module purge

project_dir=/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene
python_venv_dir=${VENV_DIR:-/fhi/home/ngoen/software/biASE-venv}

mkdir -p "${project_dir}/outputs" "${project_dir}/logs"

model_key=${MODEL_KEY:?MODEL_KEY must be set, e.g. off/omol/mh1/polar}
system_names=(cis trans)
system_name=${system_names[$SLURM_ARRAY_TASK_ID]}
run_label_base=raccoon_${model_key}_job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}
scratch_dir=/scratch/ngoen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

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
rsync -a \
  --exclude logs \
  --exclude outputs \
  --exclude 'ngoen_*' \
  "${project_dir}/" .

source "${python_venv_dir}/bin/activate"

CUDA_VISIBLE_DEVICES=0 python run_metad.py \
  --system "${system_name}" \
  --model-key "${model_key}" \
  --cv-mode 1d \
  --run-label "${run_label_base}_gpu0" &
pid1=$!

CUDA_VISIBLE_DEVICES=1 python run_metad.py \
  --system "${system_name}" \
  --model-key "${model_key}" \
  --cv-mode 2d \
  --run-label "${run_label_base}_gpu1" &
pid2=$!

sleep 5
nvidia-smi > "logs/gpu_util_${run_label_base}.txt" 2>&1 || true

wait $pid1
rc1=$?
wait $pid2
rc2=$?

[ $rc1 -ne 0 ] && echo "ERROR: GPU0 python (${system_name} 1d) exited with code ${rc1}" >&2
[ $rc2 -ne 0 ] && echo "ERROR: GPU1 python (${system_name} 2d) exited with code ${rc2}" >&2

[ $((rc1 + rc2)) -ne 0 ] && exit 1
exit 0
