#!/bin/bash
# Submit umbrella MD jobs for all new model families (polar-s/m/l, mh1)
# using the cueq-enabled venv and 12 h wall time.
# Run from the malonaldehyd directory.

mkdir -p logs

echo "=== Submitting new-model UMD jobs for malonaldehyd ==="

MODEL_FAMILY=polar MODEL_SIZE=s \
  sbatch --job-name=umd_polar_s run_U_MDs_cueq.sh
echo "  polar-s submitted"

MODEL_FAMILY=polar MODEL_SIZE=m \
  sbatch --job-name=umd_polar_m run_U_MDs_cueq.sh
echo "  polar-m submitted"

MODEL_FAMILY=polar MODEL_SIZE=l \
  sbatch --job-name=umd_polar_l run_U_MDs_cueq.sh
echo "  polar-l submitted"

MODEL_FAMILY=mh1 MODEL_SIZE=mh-1 \
  sbatch --job-name=umd_mh1 run_U_MDs_cueq.sh
echo "  mh1 submitted"

echo "=== Done. Check job IDs above and update tracking.md ==="
