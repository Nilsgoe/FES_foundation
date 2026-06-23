"""
Extract energies, forces, and coordinates from 2400 ORCA wB97M-V/def2-TZVPD
single-point calculations and write a single extended XYZ (extxyz) file.

Units in output:
  energy  : eV
  forces  : eV/Å  (= negative gradient in Eh/Bohr, converted)
  positions: Å     (as written by ORCA)

Usage
-----
  python extract_results.py \
      --base /viper/ptmp1/ngoen/Documents/azobenzene_dft \
      --outfile azobenzene_dft.xyz

The script skips any frame whose engrad file is missing or malformed and
prints a summary at the end.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

# ── Constants ─────────────────────────────────────────────────────────────────
EV_PER_EH = 27.211386246          # 1 Hartree in eV
ANG_PER_BOHR = 0.529177210903     # 1 Bohr in Å
FORCE_CONV = EV_PER_EH / ANG_PER_BOHR   # Eh/Bohr → eV/Å

# ASE atomic number → element symbol mapping (only what's needed for azobenzene)
_Z_TO_SYM = {1: "H", 6: "C", 7: "N"}

METHODS = ["off", "omol", "mh1", "polar", "pet_spice", "so3lr"]
ISOMERS = ["cis", "trans"]
N_FRAMES = 200


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_engrad(path: Path) -> tuple[list[str], np.ndarray, float, np.ndarray]:
    """
    Parse an ORCA .engrad file (ORCA 6 format).

    Returns
    -------
    symbols   : list[str]  — element symbols
    positions : (N, 3) Å   — Cartesian positions converted from Bohr
    energy    : float Eh
    gradient  : (N, 3) Eh/Bohr
    """
    # ── Identify section boundaries ──
    # Sections are separated by comment blocks ("# ... #").
    # Order in file: n_atoms | energy | gradient | coordinates
    lines = path.read_text().splitlines()

    section = None        # "natoms" | "energy" | "gradient" | "coords"
    n_atoms = None
    energy = None
    grad_vals: list[float] = []
    coord_rows: list[list[float]] = []
    symbols: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            low = stripped.lower()
            if "number of atoms" in low:
                section = "natoms"
            elif "total energy" in low:
                section = "energy"
            elif "gradient" in low and "coordinate" not in low:
                section = "gradient"
            elif "coordinate" in low:
                section = "coords"
            continue

        if section == "natoms":
            n_atoms = int(stripped)
            section = None
        elif section == "energy":
            energy = float(stripped)
            section = None
        elif section == "gradient":
            grad_vals.append(float(stripped))
            if n_atoms is not None and len(grad_vals) == 3 * n_atoms:
                section = None
        elif section == "coords":
            parts = stripped.split()
            symbols.append(_Z_TO_SYM[int(parts[0])])
            coord_rows.append([float(x) for x in parts[1:4]])

    if energy is None or not grad_vals or not coord_rows:
        raise ValueError(f"Incomplete engrad file: {path}")

    positions = np.array(coord_rows) * ANG_PER_BOHR  # Bohr → Å
    gradient = np.array(grad_vals).reshape(n_atoms, 3)
    return symbols, positions, energy, gradient


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="/viper/ptmp1/ngoen/Documents/azobenzene_dft",
        help="Root of the DFT output tree",
    )
    parser.add_argument(
        "--outfile",
        default="azobenzene_dft.xyz",
        help="Output extended XYZ filename",
    )
    args = parser.parse_args()

    base = Path(args.base)
    outfile = Path(args.outfile)

    frames = []
    n_ok = 0
    n_fail = 0

    for method in METHODS:
        for isomer in ISOMERS:
            for i in range(N_FRAMES):
                frame_name = f"frame_{i:03d}"
                calc_dir = base / "orca_calcs" / method / isomer / frame_name
                engrad_path = calc_dir / "input.engrad"

                if not engrad_path.exists():
                    print(f"MISSING  {method}/{isomer}/{frame_name}", file=sys.stderr)
                    n_fail += 1
                    continue

                try:
                    symbols, positions, energy_eh, gradient = parse_engrad(engrad_path)
                except Exception as exc:
                    print(f"ERROR    {method}/{isomer}/{frame_name}: {exc}", file=sys.stderr)
                    n_fail += 1
                    continue

                energy_ev = energy_eh * EV_PER_EH
                # forces = −gradient; convert Eh/Bohr → eV/Å
                forces = -gradient * FORCE_CONV

                atoms = Atoms(symbols=symbols, positions=positions)
                atoms.info["energy"] = energy_ev
                atoms.info["free_energy"] = energy_ev
                atoms.info["config_type"] = f"{method}_{isomer}"
                atoms.info["frame_index"] = i
                atoms.arrays["forces"] = forces

                frames.append(atoms)
                n_ok += 1

    write(str(outfile), frames, format="extxyz")
    print(f"\nWrote {n_ok} frames to {outfile}  ({n_fail} failed/missing)")


if __name__ == "__main__":
    main()
