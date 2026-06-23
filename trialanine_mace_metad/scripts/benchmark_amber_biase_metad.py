#!/usr/bin/env python
"""Benchmark AmberTools sander + biASE WT-MetaD overhead for solvated trialanine."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import sander
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from Metadynamics import WT_Metadynamics


PROJECT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/trialanine_mace_metad")
CV_INDICES = {
    "phi": (14, 16, 18, 24),
    "psi": (16, 18, 24, 26),
}


class SanderCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, prmtop: Path, restart: Path, cutoff: float = 8.0):
        super().__init__()
        self.prmtop = str(prmtop)
        self.restart = str(restart)
        self.cutoff = cutoff
        self.context = None
        self.conversion = units.kcal / units.mol
        self.calls = 0
        self.force_time_s = 0.0

    def _options(self):
        opts = sander.pme_input()
        if hasattr(opts, "cut"):
            opts.cut = self.cutoff
        return opts

    def _ensure_context(self):
        if self.context is None:
            self.context = sander.setup(self.prmtop, self.restart, None, self._options())

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._ensure_context()

        t0 = time.perf_counter()
        self.context.positions = np.asarray(self.atoms.get_positions(), dtype=float)
        self.context.box = np.asarray(self.atoms.cell.cellpar(), dtype=float)
        energy_terms, forces = self.context.energy_forces()
        self.force_time_s += time.perf_counter() - t0
        self.calls += 1

        self.results["energy"] = float(energy_terms.tot) * self.conversion
        self.results["forces"] = np.asarray(forces, dtype=float) * self.conversion

    def close(self):
        if self.context is not None:
            sander.cleanup()
            self.context = None


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
    return jnp.degrees(jnp.arctan2(jnp.dot(jnp.cross(n1, n2), b2_unit), jnp.dot(n1, n2)))


def phi_psi(positions):
    return jnp.array(
        [
            compute_dihedral(positions, CV_INDICES["phi"]),
            compute_dihedral(positions, CV_INDICES["psi"]),
        ]
    )


def restart_coordinates(prmtop: Path, restart: Path):
    opts = sander.pme_input()
    context = sander.setup(str(prmtop), str(restart), None, opts)
    positions = np.asarray(context.positions, dtype=float).copy()
    box = np.asarray(context.box, dtype=float).copy()
    sander.cleanup()
    return positions, box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Amber+sander biASE MetaD.")
    parser.add_argument("--steps", type=int, default=120, help="Use at least 101 to deposit one Gaussian at interval 100.")
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--loginterval", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=293.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--trajectory", action="store_true", help="Write a trajectory; default avoids trajectory I/O.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prmtop = PROJECT / "structures" / "trialanine_solution.prmtop"
    restart = PROJECT / "structures" / "trialanine_solution_equil_npt.rst7"
    extxyz = PROJECT / "structures" / "trialanine_solution_amber_start.extxyz"

    atoms = read(extxyz)
    pos, box = restart_coordinates(prmtop, restart)
    atoms.set_positions(pos)
    atoms.set_cell(box, scale_atoms=False)
    atoms.pbc = True

    calc = SanderCalculator(prmtop, restart)
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)

    if args.output_dir is None:
        tmpdir_obj = tempfile.TemporaryDirectory(prefix="amber_biase_bench_")
        output_dir = Path(tmpdir_obj.name)
    else:
        tmpdir_obj = None
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    bias_file = output_dir / "bench.bias"
    traj_file = output_dir / "bench.traj"

    dyn = WT_Metadynamics(
        atoms=atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=0.1,
        trajectory=str(traj_file) if args.trajectory else None,
        fixcm=False,
        cvs=phi_psi,
        bias_height=0.1,
        interval_size=args.interval,
        output_file=str(bias_file),
        well_temp=True,
        bias_factor=10,
        append_trajectory=False,
        input_file=None,
        max_bias=int(1e6),
        std_dev=[5.0, 5.0],
        wrapping=[True, True],
        bounds=((-180.0, 180.0), (-180.0, 180.0)),
        loginterval=args.loginterval,
    )

    t0 = time.perf_counter()
    dyn.run(args.steps)
    wall_s = time.perf_counter() - t0
    calc.close()

    bias_rows = 0
    if bias_file.exists():
        for line in bias_file.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("!") and not stripped.lower().startswith("time"):
                bias_rows += 1

    print(f"steps {args.steps}")
    print(f"interval {args.interval}")
    print(f"loginterval {args.loginterval}")
    print(f"trajectory_written {args.trajectory}")
    print(f"output_dir {output_dir}")
    print(f"bias_rows {bias_rows}")
    print(f"calculator_calls {calc.calls}")
    print(f"wall_s {wall_s:.9f}")
    print(f"wall_per_step_s {wall_s / args.steps:.9f}")
    print(f"steps_per_s {args.steps / wall_s:.6f}")
    print(f"sander_force_time_s {calc.force_time_s:.9f}")
    print(f"sander_force_per_call_s {calc.force_time_s / calc.calls:.9f}")
    print(f"non_sander_overhead_s {wall_s - calc.force_time_s:.9f}")
    print(f"non_sander_overhead_per_step_s {(wall_s - calc.force_time_s) / args.steps:.9f}")
    print(f"estimated_1ns_hours {wall_s / args.steps * 2_000_000 / 3600:.6f}")

    if tmpdir_obj is not None:
        tmpdir_obj.cleanup()


if __name__ == "__main__":
    main()
