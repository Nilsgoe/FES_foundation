#!/usr/bin/env python
"""Convert Amber topology/restart files to ASE extended XYZ for MACE.

Amber is used only to construct and briefly equilibrate starting structures.
The generated extxyz files are the handoff point to MACE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import parmed as pmd
from ase import Atoms
from ase.io import write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", required=True, type=Path)
    parser.add_argument("--restart", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pbc", action="store_true", help="Preserve periodic cell from Amber restart.")
    return parser.parse_args()


def atom_symbol(atom: pmd.Atom) -> str:
    if atom.element_name:
        return atom.element_name.strip().title()
    if atom.atomic_number:
        from ase.data import chemical_symbols

        return chemical_symbols[atom.atomic_number]
    raise ValueError(f"Cannot infer element for Amber atom {atom.idx} ({atom.name})")


def cell_from_box(box: np.ndarray | None) -> np.ndarray | None:
    if box is None:
        return None
    box = np.asarray(box, dtype=float).ravel()
    if len(box) < 3:
        return None
    if len(box) >= 6:
        a, b, c, alpha, beta, gamma = box[:6]
        from ase.geometry import cellpar_to_cell

        return cellpar_to_cell([a, b, c, alpha, beta, gamma])
    return np.diag(box[:3])


def main() -> None:
    args = parse_args()
    structure = pmd.load_file(str(args.prmtop), str(args.restart))

    symbols = [atom_symbol(atom) for atom in structure.atoms]
    positions = np.asarray(structure.coordinates, dtype=float)
    atoms = Atoms(symbols=symbols, positions=positions)

    if args.pbc:
        cell = cell_from_box(getattr(structure, "box", None))
        if cell is None:
            raise ValueError("PBC requested, but no Amber box vectors were found.")
        atoms.set_cell(cell)
        atoms.set_pbc(True)
        atoms.wrap(eps=1e-12)
    else:
        atoms.set_pbc(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, atoms, format="extxyz")
    print(f"Wrote {args.output} with {len(atoms)} atoms; pbc={atoms.pbc.tolist()}")


if __name__ == "__main__":
    main()
