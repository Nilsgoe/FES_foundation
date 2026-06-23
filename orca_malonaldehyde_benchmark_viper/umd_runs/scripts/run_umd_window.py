from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation


ORCA_BIN = os.environ.get("ORCA_BIN", "/mpcdf/soft/RHEL_9/packages/x86_64/orca/6.1.1/bin/orca")
CV_ATOMS = (3, 8, 4)  # CV = d(O_right-H_transfer) - d(O_left-H_transfer), zero-based ASE indices.


class DistanceDifferenceUmbrella(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, center: float, kappa: float, atoms_cv: tuple[int, int, int] = CV_ATOMS):
        super().__init__()
        self.center = float(center)
        self.kappa = float(kappa)
        self.atoms_cv = atoms_cv

    @staticmethod
    def _distance_gradient(positions: np.ndarray, atom_a: int, atom_b: int) -> tuple[float, np.ndarray, np.ndarray]:
        vector = positions[atom_a] - positions[atom_b]
        distance = float(np.linalg.norm(vector))
        if distance < 1.0e-12:
            raise RuntimeError(f"Zero distance for CV atoms {atom_a}, {atom_b}")
        grad_a = vector / distance
        grad_b = -grad_a
        return distance, grad_a, grad_b

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.get_positions()
        oxygen_right, hydrogen, oxygen_left = self.atoms_cv

        d_right, grad_or, grad_hr = self._distance_gradient(positions, oxygen_right, hydrogen)
        d_left, grad_ol, grad_hl = self._distance_gradient(positions, oxygen_left, hydrogen)
        cv = d_right - d_left
        delta = cv - self.center

        grad_cv = np.zeros_like(positions)
        grad_cv[oxygen_right] += grad_or
        grad_cv[hydrogen] += grad_hr
        grad_cv[oxygen_left] -= grad_ol
        grad_cv[hydrogen] -= grad_hl

        energy = 0.5 * self.kappa * delta * delta
        forces = -self.kappa * delta * grad_cv
        self.results = {"energy": energy, "forces": forces}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORCA/ASE umbrella MD window for malonaldehyde.")
    parser.add_argument("--window-id", type=int, required=True)
    parser.add_argument("--windows-csv", type=Path, default=Path("windows.csv"))
    parser.add_argument("--xyz", type=Path, default=Path("malonaldehyde.xyz"))
    parser.add_argument("--output-root", type=Path, default=Path("canary"))
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--friction-fs-inv", type=float, default=0.01)
    parser.add_argument("--write-interval", type=int, default=10)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260513)
    return parser.parse_args()


def read_window(path: Path, window_id: int) -> dict[str, float | int]:
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if int(row["window_id"]) == window_id:
                return {
                    "window_id": int(row["window_id"]),
                    "shift": int(row["shift"]),
                    "center_A": float(row["center_A"]),
                    "kappa_eV_A2": float(row["kappa_eV_A2"]),
                }
    raise ValueError(f"Window {window_id} not found in {path}")


def current_cv(atoms) -> float:
    oxygen_right, hydrogen, oxygen_left = CV_ATOMS
    return atoms.get_distance(oxygen_right, hydrogen) - atoms.get_distance(oxygen_left, hydrogen)


def main() -> None:
    args = parse_args()
    window = read_window(args.windows_csv, args.window_id)
    run_dir = args.output_root / f"window_{args.window_id:03d}_shift_{window['shift']:+03d}"
    orca_dir = run_dir / "orca_work"
    run_dir.mkdir(parents=True, exist_ok=True)
    orca_dir.mkdir(parents=True, exist_ok=True)

    atoms = read(args.xyz)
    rng = np.random.default_rng(args.seed + args.window_id)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_k, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    orca = ORCA(
        profile=OrcaProfile(command=ORCA_BIN),
        directory=orca_dir,
        charge=0,
        mult=1,
        orcasimpleinput="wB97M-V def2-TZVPD EnGrad",
        orcablocks=f"%pal nprocs {args.cores} end",
    )
    umbrella = DistanceDifferenceUmbrella(center=window["center_A"], kappa=window["kappa_eV_A2"])
    atoms.calc = SumCalculator([orca, umbrella])

    traj_path = run_dir / "trajectory.traj"
    xyz_path = run_dir / "trajectory.xyz"
    csv_path = run_dir / "cv_energy.csv"
    meta_path = run_dir / "metadata.txt"

    with meta_path.open("w") as handle:
        handle.write(f"window_id={window['window_id']}\n")
        handle.write(f"shift={window['shift']}\n")
        handle.write(f"center_A={window['center_A']}\n")
        handle.write(f"kappa_eV_A2={window['kappa_eV_A2']}\n")
        handle.write(f"cv_atoms_zero_based={CV_ATOMS}\n")
        handle.write("cv_definition=d(O_right-H_transfer)-d(O_left-H_transfer)\n")
        handle.write("method=wB97M-V/def2-TZVPD EnGrad\n")
        handle.write(f"orca_bin={ORCA_BIN}\n")
        handle.write(f"cores={args.cores}\n")
        handle.write(f"steps={args.steps}\n")
        handle.write(f"timestep_fs={args.timestep_fs}\n")
        handle.write(f"temperature_K={args.temperature_k}\n")

    dyn = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature_k,
        friction=args.friction_fs_inv / units.fs,
    )

    write_header = not csv_path.exists()
    start = time.perf_counter()
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["step", "time_fs", "cv_A", "potential_eV", "temperature_K", "elapsed_s"])

        def record(step: int) -> None:
            epot = atoms.get_potential_energy()
            temp = atoms.get_temperature()
            cv = current_cv(atoms)
            elapsed = time.perf_counter() - start
            writer.writerow([step, step * args.timestep_fs, cv, epot, temp, elapsed])
            handle.flush()
            if step % args.write_interval == 0:
                write(traj_path, atoms, append=True)
                write(xyz_path, atoms, append=True)

        record(0)
        for step in range(1, args.steps + 1):
            dyn.run(1)
            record(step)

    write(run_dir / "final.xyz", atoms)


if __name__ == "__main__":
    main()
