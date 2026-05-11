#!/bin/bash
set -euo pipefail

root=/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE
log_file="${root}/resubmit_metad_when_ready.log"

echo "[$(date -u)] watcher started" >> "${log_file}"

while pgrep -f 'stage_metad_restart.py' >/dev/null 2>&1; do
  echo "[$(date -u)] waiting for stage_metad_restart.py jobs to finish" >> "${log_file}"
  sleep 60
done

echo "[$(date -u)] copy stage finished, starting submissions" >> "${log_file}"

submit_if_missing() {
  local job_name=$1
  shift
  if squeue -u ngoen -h -o '%j' | grep -Fx "${job_name}" >/dev/null 2>&1; then
    echo "[$(date -u)] skip existing queued/running job ${job_name}" >> "${log_file}"
    return 0
  fi
  "$@" >> "${log_file}" 2>&1
}

submit_if_missing metad_off_small_medium_cont \
  bash -lc "cd ${root}/malonaldehyd/ngoen_26984 && sbatch run_continue_off_small_medium.sh"

submit_if_missing metad_large_models_cont \
  bash -lc "cd ${root}/f-malonaldehyd/ngoen_27329 && sbatch run_continue_large_models.sh"

submit_if_missing metad_off_small_medium_cont \
  bash -lc "cd ${root}/f-malonaldehyd/ngoen_27330 && sbatch run_continue_off_small_medium.sh"

for model_key in off omol mh1 polar; do
  submit_if_missing "metad_${model_key}_cont" \
    bash -lc "cd ${root}/azobenzene && sbatch --job-name=metad_${model_key}_cont --export=ALL,MODEL_KEY=${model_key} run_metad_continue_raccoon.sh"
done

echo "[$(date -u)] watcher finished submissions" >> "${log_file}"
