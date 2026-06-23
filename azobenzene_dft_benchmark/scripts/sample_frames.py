#!/usr/bin/env python3
"""
Randomly sample 200 frames per azobenzene MetaD trajectory and write as plain XYZ.

Usage (on Viper login node, /nexus must be mounted):
    module load python-waterboa/2024.06
    python sample_frames.py --outdir /ptmp/ngoen/azobenzene_dft/sampled_frames

Outputs:
    <outdir>/<method>/<isomer>/frame_<NNN>.xyz   (12 × 200 = 2400 files)
"""

import argparse
import random
from pathlib import Path

from ase.io import write
from ase.io.trajectory import Trajectory

# ------------------------------------------------------------------
# Trajectory paths
# MACE models: accessible via /nexus (NFS-mounted on Viper)
# PET-SPICE / SO3LR: UPDATE these two blocks before running
# ------------------------------------------------------------------

_NEXUS = "/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene/outputs"

TRAJ_PATHS: dict[tuple[str, str], str] = {
    ("off",   "cis"):  f"{_NEXUS}/metad_azob_cis_off_2d_raccoon_off_job33975_task0_gpu1.traj",
    ("off",   "trans"): f"{_NEXUS}/metad_azob_trans_off_2d_raccoon_off_job33975_task1_gpu1.traj",
    ("omol",  "cis"):  f"{_NEXUS}/metad_azob_cis_omol_2d_raccoon_omol_job33976_task0_gpu1.traj",
    ("omol",  "trans"): f"{_NEXUS}/metad_azob_trans_omol_2d_raccoon_omol_job33976_task1_gpu1.traj",
    ("mh1",   "cis"):  f"{_NEXUS}/metad_azob_cis_mh1_2d_raccoon_mh1_job33030_task0_gpu1.traj",
    ("mh1",   "trans"): f"{_NEXUS}/metad_azob_trans_mh1_2d_raccoon_mh1_job33030_task1_gpu1.traj",
    ("polar", "cis"):  f"{_NEXUS}/metad_azob_cis_polar_2d_raccoon_polar_job33031_task0_gpu1.traj",
    ("polar", "trans"): f"{_NEXUS}/metad_azob_trans_polar_2d_raccoon_polar_job33031_task1_gpu1.traj",
    ("pet_spice", "cis"):
        "/work/gpuviper_ptmp/Enhanced_sampling/upet/azobenzene/pet_spice_cis_2d/rollback_backup_2026-06-02_minus100/pet_spice_azob_cis_2d.original_before_minus100.traj",
    ("pet_spice", "trans"):
        "/work/gpuviper_ptmp/Enhanced_sampling/upet/azobenzene/pet_spice_trans_2d/outputs/pet_spice_azob_trans_2d.traj",
    ("so3lr", "cis"):
        "/work/gpuviper_ptmp/Enhanced_sampling/sol3r/azobenzene/cis_2d/outputs/sol3r_azob_cis_2d.traj",
    ("so3lr", "trans"):
        "/work/gpuviper_ptmp/Enhanced_sampling/sol3r/azobenzene/trans_2d/outputs/sol3r_azob_trans_2d.traj",
}

N_FRAMES = 200
SEED = 42


def sample_trajectory(path: str, n: int, rng: random.Random) -> list:
    """Read an ASE trajectory and return a list of (original_index, Atoms) pairs.

    Handles live/growing trajectories: a truncated final frame is silently
    dropped if it raises an exception.
    """
    traj = Trajectory(path, "r")
    n_total = len(traj)
    indices = sorted(rng.sample(range(n_total), min(n, n_total)))
    frames = []
    for i in indices:
        try:
            frames.append((i, traj[i]))
        except Exception:
            pass  # skip any corrupt trailing frame
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        required=True,
        help="Root output directory, e.g. /ptmp/ngoen/azobenzene_dft/sampled_frames",
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)

    rng = random.Random(SEED)

    header = f"{'method':<12} {'isomer':<8} {'in traj':>9} {'sampled':>9} {'skipped':>9}"
    print(header)
    print("-" * len(header))

    for (method, isomer), traj_path in TRAJ_PATHS.items():
        dest = outdir / method / isomer
        if not Path(traj_path).exists():
            print(f"{'[MISSING]':<12} {method}/{isomer}  →  {traj_path}")
            continue

        traj = Trajectory(traj_path, "r")
        n_total = len(traj)

        frames = sample_trajectory(traj_path, N_FRAMES, rng)
        n_written = 0
        dest.mkdir(parents=True, exist_ok=True)
        for seq_idx, (orig_idx, atoms) in enumerate(frames):
            out_path = dest / f"frame_{seq_idx:03d}.xyz"
            # Store original trajectory index in the comment line for traceability
            atoms.info["traj_frame"] = orig_idx
            write(str(out_path), atoms, format="xyz")
            n_written += 1

        skipped = len(frames) - n_written
        print(f"{method:<12} {isomer:<8} {n_total:>9} {n_written:>9} {skipped:>9}")

    print()
    print(f"Done. Frames written to: {outdir}")


if __name__ == "__main__":
    main()
