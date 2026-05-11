#!/bin/bash
set -euo pipefail

source ./config.env
sbatch --job-name="${JOB_NAME}" run_polar_npt_continue_viper.sh
