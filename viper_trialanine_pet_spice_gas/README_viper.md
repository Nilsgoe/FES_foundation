# Trialanine UPET/SO3LR on viper-gpu

This directory mirrors the azobenzene viper layout for trialanine.

The Amber setup was already completed on raccoon/nexus in:

`/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/trialanine_mace_metad`

Amber is not rerun here. The copied `extxyz` files are only starting structures.

## Files

- `run_trialanine.py`: shared UPET/SO3LR solution equilibration and 2D phi/psi MetaD driver.
- `run_equil_viper.sh`: solution NVT/NPT launcher.
- `run_metad_viper.sh`: restartable MetaD chunk launcher.
- `submit_chain.sh`: submits 10 dependent chunks by default with `afterany`.
- `validate_cv_indices.py`: validates phi/psi indices against gas and solution Amber PDBs.

## Defaults

- Phi: `(14, 16, 18, 24)`
- Psi: `(16, 18, 24, 26)`
- MetaD: `300 K`, `0.5 fs`, `1,000,000` steps per chunk, 10 chunks, WT bias factor `10`.
- Solution equilibration: `100000` NVT steps and `500000` NPT steps.

Production chains use `afterany`, as requested. This means later chunks start even if the previous chunk fails.
