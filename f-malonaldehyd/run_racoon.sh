#!/bin/bash -l
#SBATCH --job-name=fmalon-MetaD-run
#SBATCH --no-requeue
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH -o ./logs/job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -t 09:00:00

# Define project parameters
project_dir=/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/f-malonaldehyd
python_venv_dir=/fhi/home/ngoen/software/biASE-venv
echo $PYTHONPATH
# Copy inputs to /scratch
cd /scratch
mkdir -p /scratch/ngoen_$SLURM_JOB_ID		# This creates a unique directory for you and your job
cd ngoen_$SLURM_JOB_ID
mkdir outputs
cp $project_dir/* .

# Load your venv
source $python_venv_dir/bin/activate

# Start your calculation
python3 run_MetaD.py --model-size large 

# Copy output back to project dir
cp -r ./outputs $project_dir

# Remove files from /scratch
cd /scratch
rm -r <your username>_$SLURM_JOB_ID

# Go home (not necessary)
cd ~

