# Trialanine MACE MetaD Workflow

This workflow reproduces a capped alanine peptide setup with strict separation between AmberTools setup and MACE production physics.

AmberTools is used only to build, solvate, minimize, heat, and briefly equilibrate starting structures. All gas-phase and explicit-water production simulations, including the solvent density/box relaxation, are run with MACE and the in-repo biASE `WT_Metadynamics` implementation.

Target molecule: `ACE-ALA-ALA-ALA-NME` (`AcAla3NMe`).

## Layout

- `setup_amber/`: Amber `tleap` and short equilibration inputs.
- `structures/`: Amber and MACE starting/final structures.
- `conversion/`: Amber restart/topology to extended XYZ conversion.
- `mace_gas/`: gas-phase MACE MetaD driver wrapper area.
- `mace_solution_mh1/`: standard solution MACE output area.
- `mace_solution_polar/`: polar solution MACE output area.
- `scripts/`: SLURM scripts. These scripts do not submit anything by themselves.
- `logs/`, `outputs/`: runtime logs and trajectories.

## Important Separation

- AmberTools is allowed only in `scripts/run_amber_setup.slurm`.
- MACE scripts do not call `tleap`, `sander`, `pmemd`, `cpptraj`, or `ambpdb`.
- The Amber-solvated box is only an initial guess. Each MACE potential relaxes its own box by MACE NPT before production MetaD.

## Environments

- Amber setup: `/fhi/home/ngoen/software/ambertools_setup_env` from conda-forge AmberTools.
- MACE `off` and `omol`: `/fhi/home/ngoen/software/mace_cueq_venv`
- MACE `mh1`: `/fhi/home/ngoen/software/mace_cueq_venv`
- MACE `polar`: `polar`

## Current Scope

1. Submit or run `scripts/run_amber_setup.slurm` to generate corrected Amber starts and converted extxyz files for `AcAla3NMe`.
2. Run solution equilibration for `off`, `omol`, `mh1`, and `polar`; each performs MACE NVT/NPT from the corrected Amber solution start.
3. Do not submit MetaD until the corrected Amber and MACE equilibration stages have completed and been validated.

AmberTools 24 on `fhi-raccoon` does not ship `leaprc.protein.ff99SB`, so setup uses `leaprc.protein.ff14SB` as the closest available Amber protein force field.
