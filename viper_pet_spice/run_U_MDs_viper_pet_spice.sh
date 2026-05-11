#!/bin/bash -l
#SBATCH --job-name=umd_upet_spice
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100000
#SBATCH --time=23:59:59
#SBATCH --partition=apu
#SBATCH --gres=gpu:1
#SBATCH --array=0-40
#SBATCH -D ./
#SBATCH -o logs/umd_upet_spice_%A_%a.out
#SBATCH -e logs/umd_upet_spice_%A_%a.err

module purge
module load gcc/14 rocm/6.4 python-waterboa/2024.06

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HIPRTC_COMPILE_OPTIONS_APPEND="--no-default-config"
export HF_HOME=/ptmp/ngoen/.cache/huggingface_pet_spice
export HUGGINGFACE_HUB_CACHE=/ptmp/ngoen/.cache/huggingface_pet_spice/hub

python_venv_dir=/ptmp/ngoen/biase_venv_upet

mkdir -p outputs logs

source "${python_venv_dir}/bin/activate"

shift=$((-5 + SLURM_ARRAY_TASK_ID))
echo "Task ${SLURM_ARRAY_TASK_ID}: shift=${shift}" >&2
rocm-smi > "logs/gpu_util_upet_pet_spice_task${SLURM_ARRAY_TASK_ID}.txt" 2>&1 || true

python run_U_MD_pet_spice.py "${shift}"
rc=$?
[ $rc -ne 0 ] && echo "ERROR: shift ${shift} exited with code ${rc}" >&2
exit $rc
