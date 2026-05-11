#!/usr/bin/env python3
"""Restartable solvated trialanine mh1 MetaD for viper-gpu."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import torch
from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from mace.calculators import mace_mp

from Metadynamics import WT_Metadynamics


DEFAULT_PHI = (14, 16, 18, 24)
DEFAULT_PSI = (16, 18, 24, 26)


def parse_indices(text: str) -> tuple[int, int, int, int]:
    values = tuple(int(part) for part in text.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Expected four comma-separated 0-based atom indices.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-file", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--steps-per-chunk", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=293.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phi", type=parse_indices, default=DEFAULT_PHI)
    parser.add_argument("--psi", type=parse_indices, default=DEFAULT_PSI)
    return parser.parse_args()


def compute_dihedral(positions, indices: tuple[int, int, int, int]):
    p1, p2, p3, p4 = positions[jnp.array(indices)]
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    n1 = n1 / jnp.linalg.norm(n1)
    n2 = n2 / jnp.linalg.norm(n2)
    b2_unit = b2 / jnp.linalg.norm(b2)

    cos_phi = jnp.dot(n1, n2)
    sin_phi = jnp.dot(jnp.cross(n1, n2), b2_unit)
    return jnp.degrees(jnp.arctan2(sin_phi, cos_phi))


def build_cv_function(phi: tuple[int, int, int, int], psi: tuple[int, int, int, int]):
    def cvs(positions):
        return jnp.array([compute_dihedral(positions, phi), compute_dihedral(positions, psi)])

    return cvs


def initialize_velocities(atoms, temperature: float) -> None:
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)
    ZeroRotation(atoms)


def ensure_velocities(atoms, temperature: float) -> None:
    if atoms.get_velocities() is None:
        initialize_velocities(atoms, temperature)


def metad_paths(run_name: str):
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    trajectory_path = outputs_dir / f"{run_name}.traj"
    bias_path = outputs_dir / f"{run_name}.bias"
    final_xyz = outputs_dir / f"{run_name}_chunk_last.xyz"
    return trajectory_path, bias_path, final_xyz


def load_atoms(args: argparse.Namespace, trajectory_path: Path, bias_path: Path):
    continue_run = args.chunk_id > 0 and trajectory_path.exists() and bias_path.exists()
    if args.chunk_id > 0 and not continue_run:
        raise FileNotFoundError(
            f"Missing restart files for chunk {args.chunk_id}: "
            f"{trajectory_path} and/or {bias_path}"
        )

    if continue_run:
        atoms = read(f"{trajectory_path}@-1")
    else:
        atoms = read(args.start_file)
    return atoms.copy(), continue_run


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float32)

    trajectory_path, bias_path, final_xyz = metad_paths(args.run_name)
    atoms, continue_run = load_atoms(args, trajectory_path, bias_path)
    atoms.set_pbc(True)

    atoms.calc = mace_mp(
        model="mh-1",
        head="omol",
        enable_cueq=False,
        device=args.device,
        default_dtype="float32",
    )

    if continue_run:
        ensure_velocities(atoms, args.temperature)
    else:
        initialize_velocities(atoms, args.temperature)

    dyn = WT_Metadynamics(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=0.1,
        trajectory=str(trajectory_path),
        fixcm=False,
        cvs=build_cv_function(args.phi, args.psi),
        std_dev=[5.0, 5.0],
        bias_height=0.1,
        interval_size=100,
        output_file=str(bias_path),
        well_temp=True,
        bias_factor=10,
        append_trajectory=continue_run,
        input_file=str(bias_path) if continue_run else None,
        wrapping=[True, True],
        bounds=((-180.0, 180.0), (-180.0, 180.0)),
        max_bias=1_000_000,
    )
    dyn.run(args.steps_per_chunk)
    write(final_xyz, atoms, format="extxyz")
    print(
        f"Completed {args.run_name} chunk {args.chunk_id} "
        f"for {args.steps_per_chunk} steps; continue_run={continue_run}"
    )


if __name__ == "__main__":
    main()
