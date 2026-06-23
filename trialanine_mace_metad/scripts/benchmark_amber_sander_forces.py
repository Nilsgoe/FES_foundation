#!/usr/bin/env python
"""Benchmark AmberTools sander energy/force evaluations for solvated trialanine."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import sander


PROJECT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/trialanine_mace_metad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark AmberTools sander PME force calls.")
    parser.add_argument("--prmtop", default=str(PROJECT / "structures" / "trialanine_solution.prmtop"))
    parser.add_argument("--restart", default=str(PROJECT / "structures" / "trialanine_solution_equil_npt.rst7"))
    parser.add_argument("--n-evals", type=int, default=100)
    parser.add_argument("--displacement-std", type=float, default=0.001, help="Random displacement std in Angstrom.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cutoff", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    opts = sander.pme_input()
    if hasattr(opts, "cut"):
        opts.cut = args.cutoff

    setup_t0 = time.perf_counter()
    context = sander.setup(args.prmtop, args.restart, None, opts)
    setup_s = time.perf_counter() - setup_t0

    reference_positions = np.asarray(context.positions, dtype=float).copy()
    box = np.asarray(context.box, dtype=float).copy()
    natoms = reference_positions.shape[0]

    # Warm-up: make sure PME state is initialized before timing.
    energy_terms, forces = context.energy_forces()
    warmup_energy = float(energy_terms.tot)
    warmup_force_norm = float(np.linalg.norm(forces))

    timings = []
    energies = []
    force_norms = []
    for _ in range(args.n_evals):
        displaced = reference_positions + rng.normal(scale=args.displacement_std, size=reference_positions.shape)
        t0 = time.perf_counter()
        context.positions = displaced
        context.box = box
        energy_terms, forces = context.energy_forces()
        timings.append(time.perf_counter() - t0)
        energies.append(float(energy_terms.tot))
        force_norms.append(float(np.linalg.norm(forces)))

    sander.cleanup()

    timings = np.asarray(timings)
    energies = np.asarray(energies)
    force_norms = np.asarray(force_norms)

    print(f"natoms {natoms}")
    print(f"n_evals {args.n_evals}")
    print(f"displacement_std_A {args.displacement_std}")
    print(f"cutoff_A {args.cutoff}")
    print(f"setup_s {setup_s:.6f}")
    print(f"warmup_energy_kcal_mol {warmup_energy:.12f}")
    print(f"warmup_force_norm_kcal_mol_A {warmup_force_norm:.12f}")
    print(f"mean_eval_s {timings.mean():.9f}")
    print(f"median_eval_s {np.median(timings):.9f}")
    print(f"min_eval_s {timings.min():.9f}")
    print(f"max_eval_s {timings.max():.9f}")
    print(f"std_eval_s {timings.std(ddof=1):.9f}")
    print(f"evals_per_s {1.0 / timings.mean():.6f}")
    print(f"estimated_1ns_hours_at_0p5fs {timings.mean() * 2_000_000 / 3600:.6f}")
    print(f"energy_mean_kcal_mol {energies.mean():.12f}")
    print(f"energy_std_kcal_mol {energies.std(ddof=1):.12f}")
    print(f"force_norm_mean_kcal_mol_A {force_norms.mean():.12f}")
    print(f"force_norm_std_kcal_mol_A {force_norms.std(ddof=1):.12f}")


if __name__ == "__main__":
    main()
