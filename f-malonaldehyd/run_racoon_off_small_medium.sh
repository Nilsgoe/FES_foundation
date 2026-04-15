#!/bin/bash -l
#SBATCH --job-name=metad_off_small_medium
#SBATCH --partition=gpubig
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=125000
#SBATCH --time=23:00:00
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j

module purge

project_dir=/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/f-malonaldehyd
python_venv_dir=/fhi/home/ngoen/software/biASE-venv

cd /scratch
mkdir -p /scratch/ngoen_$SLURM_JOB_ID
cd ngoen_$SLURM_JOB_ID
mkdir -p outputs
cp -r $project_dir/* .

source $python_venv_dir/bin/activate
export PYTHONPATH=/nexus/posix0/FHI-Theory/ngoen/biASE:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python run_MetaD.py --model-size small &
pid1=$!

CUDA_VISIBLE_DEVICES=1 python run_MetaD.py --model-size medium &
pid2=$!

sleep 5
nvidia-smi > gpu_util
wait $pid1 $pid2

cd ..
cp -r ngoen_$SLURM_JOB_ID $project_dir
rm -r ngoen_$SLURM_JOB_ID
cd ~
