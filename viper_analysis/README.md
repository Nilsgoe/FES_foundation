## Viper Analysis Extension

Generated from:

- `/work/gpuviper_ptmp/Enhanced_sampling/upet`
- `/work/gpuviper_ptmp/Enhanced_sampling/sol3r`

Entry point:

- [analyze_viper_systems.py](/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/analyze_viper_systems.py)

Summary manifest:

- [analysis_summary.json](/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/viper_analysis/analysis_summary.json)

Processed systems:

- `upet/malonaldehyd`
- `upet/f-malonaldehyd`
- `upet/azobenzene/{cis_1d,cis_2d,trans_1d,trans_2d}`
- `sol3r/malonaldehyd`
- `sol3r/f-malonaldehyd`
- `sol3r/azobenzene/{cis_1d,cis_2d,trans_1d,trans_2d}`

Input-file validation result:

- No missing required inputs were detected for the processed non-trialanine `upet` and `sol3r` systems.
- Umbrella-MD cases each contain `41` `mean_cv_energy_*.csv`, `41` `cv_energy_*.csv`, and `41` `umd_*.traj` files.
- Azobenzene MetaD cases each contain `1` bias file, `1` trajectory, and `1` BFGS log.

Submission:

- Submitted new non-overwriting UPET PET-SPICE array jobs on `viper-gpu`:
  - `8461560`: `/ptmp/ngoen/Documents/Enhanced_sampling/upet/malonaldehyd`
  - `8461561`: `/ptmp/ngoen/Documents/Enhanced_sampling/upet/f-malonaldehyd`
- Remote scripts used:
  - [run_U_MD_pet_spice.py](/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/viper_pet_spice/run_U_MD_pet_spice.py)
  - [run_U_MDs_viper_pet_spice.sh](/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/viper_pet_spice/run_U_MDs_viper_pet_spice.sh)

SO3LR note:

- `pet-spice` is an UPET model family, not a SO3LR model option.
- The installed Viper UPET package explicitly supports `pet-spice-s` and `pet-spice-l`.
- The existing SO3LR submission scripts are valid, but a “pet-spice SO3LR” submission is not possible because there is no corresponding SO3LR calculator/model name.
- If a fresh SO3LR rerun is desired, use the existing remote scripts:
  - `cd /ptmp/ngoen/Documents/Enhanced_sampling/sol3r/malonaldehyd && sbatch run_U_MDs_viper.sh`
  - `cd /ptmp/ngoen/Documents/Enhanced_sampling/sol3r/f-malonaldehyd && sbatch run_U_MDs_viper.sh`

Combined comparison plots:

- [malonaldehyd_combined_energy_models.png](/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/malonaldehyd/analysis/malonaldehyd_combined_energy_models.png)
- [malonaldehyd_combined_energy_models.pdf](/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/malonaldehyd/analysis/malonaldehyd_combined_energy_models.pdf)
- [f-malonaldehyd_combined_energy_models.png](/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/f-malonaldehyd/analysis/f-malonaldehyd_combined_energy_models.png)
- [f-malonaldehyd_combined_energy_models.pdf](/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/f-malonaldehyd/analysis/f-malonaldehyd_combined_energy_models.pdf)
