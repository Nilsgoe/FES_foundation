#!/bin/bash
# Remove leftover ngoen_* scratch directories from all gpubig nodes.
# Run from the login node of fhi-raccoon.

NODES=$(scontrol show partition gpubig 2>/dev/null \
  | grep -oP '(?<!\w)Nodes=\K[^\s]+')

if [ -z "$NODES" ]; then
  echo "Could not determine node list from scontrol — using hardcoded fallback."
  NODES="slurm-worker-liked-hare-[0-7]"
fi

# Expand the SLURM nodelist into individual hostnames
NODE_LIST=$(scontrol show hostnames "$NODES" 2>/dev/null)
if [ -z "$NODE_LIST" ]; then
  echo "scontrol show hostnames failed; aborting."
  exit 1
fi

echo "=== Cleaning ngoen_* from /scratch on all gpubig nodes ==="
echo "Nodes: $(echo $NODE_LIST | tr '\n' ' ')"
echo ""

for node in $NODE_LIST; do
  echo -n "[$node] "
  result=$(ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
    "$node" \
    'dirs=$(find /scratch -maxdepth 1 -name "ngoen_*" -type d 2>/dev/null)
     if [ -z "$dirs" ]; then
       echo "nothing to clean"
     else
       echo "removing: $(echo $dirs | tr "\n" " ")"
       rm -rf $dirs
     fi' 2>&1)
  echo "$result"
done

echo ""
echo "=== Done ==="
