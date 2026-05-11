#!/bin/bash
set -euo pipefail

source ./config.env

chunks="${1:-${CHUNKS}}"
steps_per_chunk="${2:-${STEPS_PER_CHUNK}}"
dependency=""

echo "Submitting ${RUN_NAME}: ${chunks} chunks x ${steps_per_chunk} steps"

for chunk in $(seq 0 $((chunks - 1))); do
    jobid=$(sbatch --parsable \
        ${dependency:+--dependency=afterok:${dependency}} \
        --job-name="${JOB_NAME}" \
        --export=ALL,CHUNK_ID="${chunk}",STEPS_PER_CHUNK_OVERRIDE="${steps_per_chunk}" \
        run_metad_viper.sh)
    echo "chunk=${chunk} jobid=${jobid} dependency=${dependency:-none}"
    dependency="${jobid}"
done
