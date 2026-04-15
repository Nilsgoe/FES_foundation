#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J MetaD_fmalon 
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --ntasks-per-node=1

#SBATCH --mail-type=none
#SBATCH --mail-user=goennheimer@fhi.mpg.de
#SBATCH --time=19:59:59

module purge
module load intel/21.2.0 impi/2021.2 cuda/11.6 #anaconda/3/2023.03

source /ptmp/ngoen/mace-biase-venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python3 run_MetaD.py --model-size large 
