#!/bin/bash -l
#SBATCH --job-name=umd_cueq
#SBATCH --partition=gpubig
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=125000
#SBATCH --time=12:00:00
#SBATCH --array=0-20%1
#SBATCH -o /nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/f-malonaldehyd/logs/umd_cueq_%x_%A_%a.out
#SBATCH -e /nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/f-malonaldehyd/logs/umd_cueq_%x_%A_%a.err

module purge

project_dir=/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/f-malonaldehyd
python_venv_dir=/work/mace_alpha_cueq_venv

# Ensure destination directories exist before SLURM tries to write logs there
mkdir -p "${project_dir}/outputs" "${project_dir}/logs"

model_family=${MODEL_FAMILY:-polar}
model_size=${MODEL_SIZE:-l}
model_tag=${model_family}_${model_size}

start_shift=$((-5 + 2 * SLURM_ARRAY_TASK_ID))
shift_a=${start_shift}
shift_b=$((start_shift + 1))

run_label_base=raccoon_${model_tag}_job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}
scratch_dir=/scratch/ngoen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

export PYTHONPATH=/nexus/posix0/FHI-Theory/ngoen/biASE:$PYTHONPATH

# Always copy all outputs and logs back, even on crash or signal.
# SLURM stderr/stdout go directly to project_dir/logs/ via the absolute
# -o/-e paths above, so we only need to rescue Python-generated files here.
cleanup() {
    echo "--- cleanup triggered: copying scratch outputs to ${project_dir} ---" >&2
    shopt -s nullglob
    for f in "${scratch_dir}/outputs"/*; do
        cp -f "${f}" "${project_dir}/outputs/" 2>&1 >&2 || true
    done
    for f in "${scratch_dir}/logs"/*; do
        cp -f "${f}" "${project_dir}/logs/" 2>&1 >&2 || true
    done
    shopt -u nullglob
    rm -rf "${scratch_dir}"
}
trap cleanup EXIT

mkdir -p "${scratch_dir}"
cd "${scratch_dir}"
mkdir -p outputs logs
# Exclude logs/ and outputs/ so the initial copy never snapshots the live
# SLURM output files — cleanup would otherwise overwrite them with empty copies.
shopt -s extglob
cp -r "${project_dir}"/!(logs|outputs|ngoen_*) .
shopt -u extglob

source "${python_venv_dir}/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CUDA_VISIBLE_DEVICES=0 python run_U_MD.py "${shift_a}" \
  --model-family "${model_family}" \
  --model-size "${model_size}" \
  --run-label "${run_label_base}_gpu0" &
pid1=$!

pid2=
if [ "${shift_b}" -le 35 ]; then
  CUDA_VISIBLE_DEVICES=1 python run_U_MD.py "${shift_b}" \
    --model-family "${model_family}" \
    --model-size "${model_size}" \
    --run-label "${run_label_base}_gpu1" &
  pid2=$!
fi

sleep 5
nvidia-smi > "logs/gpu_util_${run_label_base}.txt"

wait $pid1
rc1=$?
rc2=0
if [ -n "${pid2}" ]; then
  wait $pid2
  rc2=$?
fi

[ $rc1 -ne 0 ] && echo "ERROR: GPU0 python (shift ${shift_a}) exited with code ${rc1}" >&2
[ $rc2 -ne 0 ] && echo "ERROR: GPU1 python (shift ${shift_b}) exited with code ${rc2}" >&2

[ $((rc1 + rc2)) -ne 0 ] && exit 1
exit 0
