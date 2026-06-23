#!/usr/bin/env python
"""Azobenzene gas-phase MetaD with the fine-tuned SO3LR model.

Energies from the fine-tuned model are E0-subtracted and mean-centred (shift
= -120.05 eV).  Forces are unaffected by constant shifts, so the MetaD bias
is correct regardless.

Protocol per chunk:
  chunk 0 : BFGS minimize → NVT equil (--nvt-steps) → MetaD (--steps-per-chunk)
  chunk n>0: continue MetaD from existing trajectory/bias files
"""

from __future__ import annotations

# Patch before any orbax/so3lr imports: jax ≥0.6 renamed DeviceLocalLayout → Layout
import jax.experimental.layout as _jl
if not hasattr(_jl, "DeviceLocalLayout"):
    _jl.DeviceLocalLayout = _jl.Layout

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS

from Metadynamics import WT_Metadynamics


FT_WORKDIR = "/ptmp/ngoen/Documents/azobenzene_so3lr_training/ft_so3lr"

TEMPERATURE_K = 300.0
FRICTION = 0.1
TIMESTEP = 0.5 * units.fs
BIAS_HEIGHT = 0.1
BIAS_FACTOR = 10
INTERVAL_SIZE = 100
MAX_BIAS = 1_000_000
STD_DEV_1D = 5.0
STD_DEV_2D = [5.0, 5.0]
DIHEDRAL_BOUNDS = (-180.0, 180.0)
ANGLE_BOUNDS = (0.0, 180.0)

INDEX_MAP = {
    "cis": {
        "dihedral": (1, 6, 7, 8),
        "angle": (1, 6, 7),
        "start_file": "/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene/azob_cis_opt.traj",
    },
    "trans": {
        "dihedral": (2, 11, 12, 13),
        "angle": (2, 11, 12),
        "start_file": "/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene/azob_trans_opt.traj",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", choices=["cis", "trans"], required=True)
    p.add_argument("--cv-mode", choices=["1d", "2d"], required=True)
    p.add_argument("--chunk-id", type=int, required=True)
    p.add_argument("--steps-per-chunk", type=int, required=True)
    p.add_argument("--nvt-steps", type=int, default=40_000, help="NVT steps before MetaD (chunk 0 only).")
    p.add_argument("--start-file", default=None)
    p.add_argument(
        "--restart-offset",
        type=int,
        default=-1,
        help="Frame index for continuation (e.g. -1000 to skip a bad tail).",
    )
    p.add_argument("--trajectory-loginterval", type=int, default=10)
    return p.parse_args()


def build_calculator():
    from mlff.md import mlffCalculatorSparse

    # So3lrCalculator always loads the package-bundled pretrained params; bypass it
    # and load directly from the fine-tuned orbax checkpoint in FT_WORKDIR/checkpoints/.
    return mlffCalculatorSparse.create_from_ckpt_dir(
        ckpt_dir=FT_WORKDIR,
        from_file=False,   # False = load orbax checkpoint; True = load pretrained raw files
        calculate_stress=False,
        lr_cutoff=1000,
        lr_neighbors_bool=True,
        dispersion_energy_cutoff_lr_damping=2.0,
    )


def compute_dihedral(positions, indices):
    p1, p2, p3, p4 = positions[jnp.array(indices)]
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    n1 = n1 / jnp.linalg.norm(n1)
    n2 = n2 / jnp.linalg.norm(n2)
    b2_u = b2 / jnp.linalg.norm(b2)
    return jnp.degrees(jnp.arctan2(jnp.dot(jnp.cross(n1, n2), b2_u), jnp.dot(n1, n2)))


def compute_angle(positions, indices):
    pa, pb, pc = positions[jnp.array(indices)]
    v1 = pa - pb
    v2 = pc - pb
    cos_t = jnp.clip(jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2)), -1.0, 1.0)
    return jnp.degrees(jnp.arccos(cos_t))


def build_cvs(system: str, cv_mode: str):
    dih = INDEX_MAP[system]["dihedral"]
    ang = INDEX_MAP[system]["angle"]

    def cv_1d(pos):
        return jnp.array([compute_dihedral(pos, dih)])

    def cv_2d(pos):
        return jnp.array([compute_dihedral(pos, dih), compute_angle(pos, ang)])

    return cv_1d if cv_mode == "1d" else cv_2d


def run_nvt(atoms, nvt_steps: int, run_tag: str, outputs_dir: Path) -> None:
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)
    dyn = Langevin(
        atoms,
        timestep=TIMESTEP,
        temperature_K=TEMPERATURE_K,
        friction=FRICTION,
        trajectory=str(outputs_dir / f"{run_tag}.nvt.traj"),
        logfile=str(outputs_dir / f"{run_tag}.nvt.log"),
        loginterval=500,
    )
    dyn.run(nvt_steps)
    write(outputs_dir / f"{run_tag}.nvt_final.xyz", atoms, format="extxyz")
    print(f"NVT equilibration done ({nvt_steps} steps = {nvt_steps * 0.5 / 1000:.1f} ps)")


def main() -> None:
    args = parse_args()
    spec = INDEX_MAP[args.system]
    run_tag = f"ft_so3lr_azob_{args.system}_{args.cv_mode}"
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    traj_path = outputs_dir / f"{run_tag}.traj"
    bias_path = outputs_dir / f"{run_tag}.bias"

    continue_run = args.chunk_id > 0 and traj_path.exists() and bias_path.exists()
    if args.chunk_id > 0 and not continue_run:
        raise FileNotFoundError(
            f"Continuation (chunk {args.chunk_id}) requires {traj_path} and {bias_path}"
        )

    if continue_run:
        atoms = read(f"{traj_path}@{args.restart_offset}").copy()
    else:
        atoms = read(args.start_file or spec["start_file"]).copy()

    atoms.set_pbc(False)
    atoms.calc = build_calculator()

    if not continue_run:
        bfgs_log = outputs_dir / f"bfgs_{run_tag}.log"
        BFGS(atoms, logfile=str(bfgs_log)).run(fmax=0.05, steps=500)
        run_nvt(atoms, args.nvt_steps, run_tag, outputs_dir)
    else:
        if atoms.get_velocities() is None:
            MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)

    cv_function = build_cvs(args.system, args.cv_mode)
    metad_kwargs = dict(
        atoms=atoms,
        timestep=TIMESTEP,
        temperature_K=TEMPERATURE_K,
        friction=FRICTION,
        trajectory=str(traj_path),
        fixcm=True,
        cvs=cv_function,
        bias_height=BIAS_HEIGHT,
        interval_size=INTERVAL_SIZE,
        output_file=str(bias_path),
        well_temp=True,
        bias_factor=BIAS_FACTOR,
        append_trajectory=continue_run,
        input_file=str(bias_path) if continue_run else None,
        max_bias=MAX_BIAS,
        loginterval=args.trajectory_loginterval,
    )
    if args.cv_mode == "1d":
        metad_kwargs.update(std_dev=STD_DEV_1D, wrapping=[True], bounds=(DIHEDRAL_BOUNDS,))
    else:
        metad_kwargs.update(std_dev=STD_DEV_2D, wrapping=[True, False], bounds=(DIHEDRAL_BOUNDS, ANGLE_BOUNDS))

    dyn = WT_Metadynamics(**metad_kwargs)
    dyn.run(args.steps_per_chunk)
    write(outputs_dir / f"{run_tag}_chunk{args.chunk_id}_last.xyz", atoms, format="extxyz")
    print(
        f"Completed {run_tag} chunk {args.chunk_id} "
        f"({args.steps_per_chunk} MetaD steps; continue_run={continue_run})"
    )


if __name__ == "__main__":
    main()
