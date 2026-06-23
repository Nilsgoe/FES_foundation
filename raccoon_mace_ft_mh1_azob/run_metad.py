#!/usr/bin/env python
"""Run 2 ns azobenzene 2D WT-MetaD with the fine-tuned MACE-MH1 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS

from Metadynamics import WT_Metadynamics


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
MODEL_PATH = ROOT / "azobenzene_dft_benchmark/models/azob_ft_mh1.model"
OUTPUTS = Path("outputs")

TEMPERATURE_K = 333.0
TIMESTEP = 0.5 * units.fs
FRICTION = 0.1
BIAS_HEIGHT_EV = 0.1
BIAS_FACTOR = 10
INTERVAL_STEPS = 100
STD_DEV_DEG = [5.0, 5.0]
STEPS = 4_000_000

SYSTEMS = {
    "cis": {
        "start": ROOT / "azobenzene/azob_cis_opt.traj",
        "dihedral": (1, 6, 7, 8),
        "angle": (1, 6, 7),
    },
    "trans": {
        "start": ROOT / "azobenzene/azob_trans_opt.traj",
        "dihedral": (2, 11, 12, 13),
        "angle": (2, 11, 12),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--nvt-steps", type=int, default=40_000)
    parser.add_argument("--trajectory-loginterval", type=int, default=10)
    parser.add_argument("--continue-run", action="store_true")
    return parser.parse_args()


def build_calculator():
    from mace.calculators import MACECalculator

    return MACECalculator(
        model_paths=[str(MODEL_PATH)],
        device="cuda",
        default_dtype="float64",
        enable_cueq=False,
    )


def dihedral(positions, indices):
    p1, p2, p3, p4 = positions[jnp.array(indices)]
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    n1 /= jnp.linalg.norm(n1)
    n2 /= jnp.linalg.norm(n2)
    b2 /= jnp.linalg.norm(b2)
    return jnp.degrees(jnp.arctan2(jnp.dot(jnp.cross(n1, n2), b2), jnp.dot(n1, n2)))


def angle(positions, indices):
    pa, pb, pc = positions[jnp.array(indices)]
    v1 = pa - pb
    v2 = pc - pb
    cosine = jnp.clip(jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2)), -1.0, 1.0)
    return jnp.degrees(jnp.arccos(cosine))


def build_cvs(system: str):
    spec = SYSTEMS[system]

    def cvs(positions):
        return jnp.array(
            [
                dihedral(positions, spec["dihedral"]),
                angle(positions, spec["angle"]),
            ]
        )

    return cvs


def main() -> None:
    args = parse_args()
    spec = SYSTEMS[args.system]
    run_name = f"mace_ft_mh1_azob_{args.system}_2d"
    OUTPUTS.mkdir(exist_ok=True)

    traj_path = OUTPUTS / f"{run_name}.traj"
    bias_path = OUTPUTS / f"{run_name}.bias"

    if args.continue_run:
        if not traj_path.exists() or not bias_path.exists():
            raise FileNotFoundError(f"Missing continuation files: {traj_path} and/or {bias_path}")
        atoms = read(f"{traj_path}@-1").copy()
    else:
        atoms = read(spec["start"]).copy()
    atoms.set_pbc(False)
    atoms.calc = build_calculator()

    if not args.continue_run:
        BFGS(atoms, logfile=str(OUTPUTS / f"bfgs_{run_name}.log")).run(fmax=0.05, steps=500)
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)
        nvt = Langevin(
            atoms,
            timestep=TIMESTEP,
            temperature_K=TEMPERATURE_K,
            friction=FRICTION,
            trajectory=str(OUTPUTS / f"{run_name}.nvt.traj"),
            logfile=str(OUTPUTS / f"{run_name}.nvt.log"),
            loginterval=500,
            fixcm=True,
        )
        nvt.run(args.nvt_steps)
        write(OUTPUTS / f"{run_name}.nvt_final.xyz", atoms, format="extxyz")

    dyn = WT_Metadynamics(
        atoms=atoms,
        timestep=TIMESTEP,
        temperature_K=TEMPERATURE_K,
        friction=FRICTION,
        trajectory=str(traj_path),
        fixcm=True,
        cvs=build_cvs(args.system),
        std_dev=STD_DEV_DEG,
        bias_height=BIAS_HEIGHT_EV,
        interval_size=INTERVAL_STEPS,
        output_file=str(bias_path),
        well_temp=True,
        bias_factor=BIAS_FACTOR,
        wrapping=[True, False],
        bounds=((-180.0, 180.0), (0.0, 180.0)),
        max_bias=1_000_000,
        loginterval=args.trajectory_loginterval,
        append_trajectory=args.continue_run,
        input_file=str(bias_path) if args.continue_run else None,
    )
    dyn.run(args.steps)
    write(OUTPUTS / f"{run_name}_last.xyz", atoms, format="extxyz")


if __name__ == "__main__":
    main()
