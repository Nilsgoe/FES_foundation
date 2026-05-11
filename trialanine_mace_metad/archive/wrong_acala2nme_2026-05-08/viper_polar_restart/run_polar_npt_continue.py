#!/usr/bin/env python3
"""Continue the solvated trialanine polar NPT run on viper-gpu."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from mace.calculators.mace import MACECalculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-traj", type=Path, required=True)
    parser.add_argument("--fallback-start", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--npt-output", type=Path, required=True)
    parser.add_argument("--npt-steps", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=293.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--default-dtype", default="float32")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_last_valid_traj_frame(path: Path):
    traj = Trajectory(path)
    for idx in range(len(traj) - 1, -1, -1):
        try:
            atoms = traj[idx].copy()
            return atoms, idx
        except Exception:
            continue
    raise ValueError(f"No readable frame found in trajectory {path}")


def load_restart_atoms(input_traj: Path, fallback_start: Path):
    if input_traj.exists() and input_traj.stat().st_size > 0:
        try:
            atoms, idx = load_last_valid_traj_frame(input_traj)
            return atoms, f"{input_traj}@{idx}"
        except Exception as exc:
            print(f"Failed to read a valid frame from {input_traj}: {exc}")
    atoms = read(fallback_start).copy()
    return atoms, str(fallback_start)


def ensure_velocities(atoms, temperature: float) -> None:
    if atoms.get_velocities() is None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        Stationary(atoms)
        ZeroRotation(atoms)


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float32)

    atoms, source = load_restart_atoms(args.input_traj, args.fallback_start)
    atoms.set_pbc(True)
    if atoms.cell.volume <= 0:
        raise ValueError(f"Restart structure from {source} has no periodic cell.")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.npt_output.parent.mkdir(parents=True, exist_ok=True)

    atoms.calc = MACECalculator(
        model_paths=str(args.model_path),
        device=args.device,
        default_dtype=args.default_dtype,
        model_type="PolarMACE",
    )
    ensure_velocities(atoms, args.temperature)

    print(f"Restart source: {source}")
    print(f"Atom count: {len(atoms)}")
    print(f"Cell volume: {atoms.cell.volume:.6f} A^3")
    print(f"NPT steps: {args.npt_steps}")

    dyn = NPTBerendsen(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        pressure_au=args.pressure_bar * units.bar,
        taut=1000.0 * units.fs,
        taup=2000.0 * units.fs,
        compressibility_au=4.57e-5 / units.bar,
        trajectory=str(args.output_prefix.with_suffix(".npt.traj")),
        logfile=str(args.output_prefix.with_suffix(".npt.log")),
        loginterval=100,
    )
    dyn.run(args.npt_steps)

    write(args.npt_output, atoms, format="extxyz")
    write(args.output_prefix.with_suffix(".npt_final.xyz"), atoms, format="extxyz")
    print(f"Wrote final structure to {args.npt_output}")


if __name__ == "__main__":
    main()
