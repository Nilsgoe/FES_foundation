#!/bin/bash -l
#SBATCH --job-name=umd_raccoon
#SBATCH --partition=gpubig
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=125000
#SBATCH --time=06:00:00
#SBATCH --array=0-20%1
#SBATCH -o ./logs/umd_raccoon_%x_%A_%a.out
#SBATCH -e ./logs/umd_raccoon_%x_%A_%a.err

module purge

project_dir=/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/malonaldehyd
python_venv_dir=/fhi/home/ngoen/software/biASE-venv

model_family=${MODEL_FAMILY:-off}
model_size=${MODEL_SIZE:-large}
model_tag=${model_family}_${model_size}

start_shift=$((-5 + 2 * SLURM_ARRAY_TASK_ID))
shift_a=${start_shift}
shift_b=$((start_shift + 1))

run_label_base=raccoon_${model_tag}_job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}
export PYTHONPATH=/nexus/posix0/FHI-Theory/ngoen/biASE:$PYTHONPATH
cd /scratch
mkdir -p /scratch/ngoen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
cd ngoen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p outputs logs
cp -r $project_dir/* .

source $python_venv_dir/bin/activate

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
if [ -n "${pid2}" ]; then
  wait $pid2
fi

mkdir -p "${project_dir}/outputs"
mkdir -p "${project_dir}/logs"

shopt -s nullglob
for file in outputs/*"${run_label_base}"*; do
  cp "${file}" "${project_dir}/outputs/"
done
for file in logs/*"${run_label_base}"*; do
  cp "${file}" "${project_dir}/logs/"
done
shopt -u nullglob

cp "logs/umd_raccoon_${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${project_dir}/logs/" 2>/dev/null || true
cp "logs/umd_raccoon_${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${project_dir}/logs/" 2>/dev/null || true

cd /scratch
rm -r "ngoen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd ~
