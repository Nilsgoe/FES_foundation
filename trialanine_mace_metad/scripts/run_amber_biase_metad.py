#!/usr/bin/env python
"""Run Amber-force-field trialanine WT-MetaD through ASE + biASE.

Amber provides only potential energy and forces through the AmberTools
``sander`` Python API. biASE provides the 2D well-tempered metadynamics driver.
No PLUMED is used.
"""

from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from Metadynamics import WT_Metadynamics

import sander


PROJECT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/trialanine_mace_metad")
CV_INDICES = {
    # 0-based ASE indices for corrected AcAla3NMe.
    "phi": (14, 16, 18, 24),
    "psi": (16, 18, 24, 26),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Amber+biASE reference MetaD for trialanine.")
    parser.add_argument("--phase", choices=("gas", "solution"), required=True)
    parser.add_argument("--steps", type=int, default=2_000_000, help="MD steps; 2,000,000 at 0.5 fs = 1 ns.")
    parser.add_argument("--temperature", type=float, default=293.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--bias-height-ev", type=float, default=0.1)
    parser.add_argument("--sigma-deg", type=float, default=5.0)
    parser.add_argument("--deposition-interval", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "outputs" / "amber_reference_metad",
        help="Directory for trajectory and bias outputs.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=PROJECT / "amber_reference_metad",
        help="Directory for per-phase sander working files.",
    )
    parser.add_argument("--trajectory-loginterval", type=int, default=10)
    parser.add_argument("--run-label", default="amber_biase_1ns")
    parser.add_argument("--restart-traj", type=Path, default=None, help="Read the last image from this ASE trajectory.")
    parser.add_argument("--input-bias", type=Path, default=None, help="Existing biASE bias file to continue.")
    parser.add_argument("--append-trajectory", action="store_true", help="Append to the trajectory instead of creating a fresh one.")
    return parser.parse_args()


def compute_dihedral(positions, indices):
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


def build_cv_function():
    def phi_psi(positions):
        return jnp.array(
            [
                compute_dihedral(positions, CV_INDICES["phi"]),
                compute_dihedral(positions, CV_INDICES["psi"]),
            ]
        )

    return phi_psi


class SanderCalculator(Calculator):
    """ASE calculator backed by AmberTools sander energy/force calls."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, prmtop: Path, restart: Path, phase: str):
        super().__init__()
        self.prmtop = str(prmtop)
        self.restart = str(restart)
        self.phase = phase
        self.context = None
        self.conversion = units.kcal / units.mol
        atexit.register(self.close)

    def _input_options(self):
        if self.phase == "solution":
            opts = sander.pme_input()
            if hasattr(opts, "cut"):
                opts.cut = 8.0
            return opts
        return sander.gas_input()

    @staticmethod
    def _box_from_atoms(atoms):
        if not atoms.pbc.any():
            return None
        return np.asarray(atoms.cell.cellpar(), dtype=float)

    def _ensure_context(self, atoms):
        if self.context is not None:
            return
        self.context = sander.setup(self.prmtop, self.restart, None, self._input_options())

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._ensure_context(self.atoms)

        self.context.positions = np.asarray(self.atoms.get_positions(), dtype=float)
        box = self._box_from_atoms(self.atoms)
        if box is not None:
            self.context.box = box

        energy_terms, forces = self.context.energy_forces()
        self.results["energy"] = float(energy_terms.tot) * self.conversion
        self.results["forces"] = np.asarray(forces, dtype=float) * self.conversion

    def close(self):
        if self.context is not None:
            sander.cleanup()
            self.context = None


def phase_paths(phase: str) -> tuple[Path, Path, Path]:
    if phase == "gas":
        return (
            PROJECT / "structures" / "trialanine_gas_amber_start.extxyz",
            PROJECT / "structures" / "trialanine_gas.prmtop",
            PROJECT / "structures" / "trialanine_gas_equil_nvt.rst7",
        )
    return (
        PROJECT / "structures" / "trialanine_solution_amber_start.extxyz",
        PROJECT / "structures" / "trialanine_solution.prmtop",
        PROJECT / "structures" / "trialanine_solution_equil_npt.rst7",
    )


def restart_coordinates(prmtop: Path, restart: Path, phase: str):
    opts = sander.pme_input() if phase == "solution" else sander.gas_input()
    context = sander.setup(str(prmtop), str(restart), None, opts)
    positions = np.asarray(context.positions, dtype=float).copy()
    box = np.asarray(context.box, dtype=float).copy()
    sander.cleanup()
    return positions, box


def main() -> None:
    args = parse_args()
    start_file, topology, restart = phase_paths(args.phase)
    run_dir = args.work_dir / args.phase / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    atoms = read(start_file)
    restart_pos, restart_box = restart_coordinates(topology, restart, args.phase)
    atoms.set_positions(restart_pos)
    if args.phase == "gas":
        atoms.pbc = False
    else:
        atoms.pbc = True
        atoms.set_cell(restart_box, scale_atoms=False)

    if args.restart_traj is not None:
        restart_atoms = read(args.restart_traj, index=-1)
        atoms.set_positions(restart_atoms.get_positions())
        atoms.set_cell(restart_atoms.get_cell(), scale_atoms=False)
        atoms.pbc = restart_atoms.pbc

    atoms.calc = SanderCalculator(topology, restart, args.phase)

    if atoms.get_momenta().shape != (len(atoms), 3) or not np.any(atoms.get_momenta()):
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)

    output_prefix = args.output_dir / f"{args.phase}_{args.run_label}"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    dyn = WT_Metadynamics(
        atoms=atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=0.1,
        trajectory=str(output_prefix.with_suffix(".traj")),
        fixcm=(args.phase == "gas"),
        cvs=build_cv_function(),
        bias_height=args.bias_height_ev,
        interval_size=args.deposition_interval,
        output_file=str(output_prefix.with_suffix(".bias")),
        well_temp=True,
        bias_factor=10,
        append_trajectory=args.append_trajectory,
        input_file=str(args.input_bias) if args.input_bias is not None else None,
        max_bias=int(1e6),
        std_dev=[args.sigma_deg, args.sigma_deg],
        wrapping=[True, True],
        bounds=((-180.0, 180.0), (-180.0, 180.0)),
        loginterval=args.trajectory_loginterval,
    )
    dyn.run(args.steps)


if __name__ == "__main__":
    main()
