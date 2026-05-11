#!/bin/bash -l
#SBATCH --job-name=metad_azob_cont
#SBATCH --partition=gpubig
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=125000
#SBATCH --time=3-23:00:00
#SBATCH --array=0-1%1
#SBATCH -o /nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene/logs/metad_azob_cont_%x_%j.out
#SBATCH -e /nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene/logs/metad_azob_cont_%x_%j.err

set -euo pipefail

module purge

project_dir=/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene
model_key=${MODEL_KEY:?MODEL_KEY must be set}

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

system_names=(cis trans)
system_name=${system_names[$SLURM_ARRAY_TASK_ID]}
scratch_dir=/scratch/ngoen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_cont

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

stage_run() {
  local cv_mode=$1
  local gpu_id=$2
  local pattern="outputs/metad_azob_${system_name}_${model_key}_${cv_mode}_*.traj"
  shopt -s nullglob
  local matches=(${pattern})
  shopt -u nullglob
  if [ ${#matches[@]} -ne 1 ]; then
    echo "Expected exactly one match for ${pattern}, found ${#matches[@]}" >&2
    return 1
  fi
  local orig_traj="${matches[0]}"
  local stem
  stem=$(basename "${orig_traj}" .traj)
  local old_run_label="${stem#metad_azob_${system_name}_${model_key}_${cv_mode}_}"

  CUDA_VISIBLE_DEVICES=${gpu_id} python run_metad.py \
    --system "${system_name}" \
    --model-key "${model_key}" \
    --cv-mode "${cv_mode}" \
    --run-label "${old_run_label}" \
    --steps 1000000 \
    --continue-run \
    --trajectory-loginterval 10
}

stage_run 1d 0 &
pid1=$!
stage_run 2d 1 &
pid2=$!

sleep 5
nvidia-smi > "logs/gpu_util_cont_${model_key}_${system_name}.txt" 2>&1 || true

wait $pid1
rc1=$?
wait $pid2
rc2=$?

[ $rc1 -ne 0 ] && echo "ERROR: GPU0 continuation failed with code ${rc1}" >&2
[ $rc2 -ne 0 ] && echo "ERROR: GPU1 continuation failed with code ${rc2}" >&2
[ $((rc1 + rc2)) -ne 0 ] && exit 1
