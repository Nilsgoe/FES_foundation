#!/usr/bin/env python
"""Validate trialanine phi/psi atom indices against an Amber PDB."""

from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED_PHI_LABELS = (("ALA", 2, "C"), ("ALA", 3, "N"), ("ALA", 3, "CA"), ("ALA", 3, "C"))
EXPECTED_PSI_LABELS = (("ALA", 3, "N"), ("ALA", 3, "CA"), ("ALA", 3, "C"), ("NME", 4, "N"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdb",
        default="structures/trialanine_gas_initial.pdb",
        type=Path,
        help="Amber PDB used to validate atom ordering.",
    )
    parser.add_argument(
        "--compare-solution-pdb",
        type=Path,
        help="Optional solution PDB. When set, verify gas/solution peptide atom ordering is identical.",
    )
    parser.add_argument("--phi", default="14,16,18,24", help="0-based comma-separated phi indices.")
    parser.add_argument("--psi", default="16,18,24,26", help="0-based comma-separated psi indices.")
    return parser.parse_args()


def parse_indices(text: str) -> tuple[int, int, int, int]:
    values = tuple(int(part) for part in text.split(","))
    if len(values) != 4:
        raise ValueError(f"Expected four comma-separated indices, got {text!r}")
    return values


def read_pdb_atoms(path: Path) -> list[dict[str, object]]:
    atoms = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atoms.append(
                {
                    "index": len(atoms),
                    "serial": int(line[6:11]),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "resid": int(line[22:26]),
                }
            )
    return atoms


def labels_for_indices(atoms: list[dict[str, object]], indices: tuple[int, int, int, int]):
    labels = []
    for idx in indices:
        atom = atoms[idx]
        labels.append((atom["resname"], atom["resid"], atom["name"]))
    return tuple(labels)


def find_expected_indices(atoms: list[dict[str, object]], expected_labels):
    indices = []
    for label in expected_labels:
        matches = [
            atom["index"]
            for atom in atoms
            if (atom["resname"], atom["resid"], atom["name"]) == label
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one atom matching {label}, found {matches}")
        indices.append(matches[0])
    return tuple(indices)


def report(name: str, actual_indices, actual_labels, expected_indices, expected_labels) -> bool:
    ok = actual_indices == expected_indices and actual_labels == expected_labels
    print(f"{name}: {'OK' if ok else 'FAIL'}")
    print(f"  actual indices:   {actual_indices}")
    print(f"  actual labels:    {actual_labels}")
    print(f"  expected indices: {expected_indices}")
    print(f"  expected labels:  {expected_labels}")
    return ok


def peptide_signature(atoms: list[dict[str, object]]):
    signature = []
    for atom in atoms:
        if atom["resname"] == "WAT":
            continue
        signature.append((atom["index"], atom["resname"], atom["resid"], atom["name"]))
    return tuple(signature)


def compare_gas_solution_order(gas_atoms, solution_atoms) -> bool:
    gas_signature = peptide_signature(gas_atoms)
    solution_signature = peptide_signature(solution_atoms)
    ok = gas_signature == solution_signature
    print(f"gas/solution peptide atom-order comparison: {'OK' if ok else 'FAIL'}")
    print(f"  gas peptide atoms:      {len(gas_signature)}")
    print(f"  solution peptide atoms: {len(solution_signature)}")
    if not ok:
        for gas_atom, solution_atom in zip(gas_signature, solution_signature):
            if gas_atom != solution_atom:
                print(f"  first mismatch: gas={gas_atom}, solution={solution_atom}")
                break
        if len(gas_signature) != len(solution_signature):
            print("  peptide atom counts differ")
    return ok


def main() -> None:
    args = parse_args()
    atoms = read_pdb_atoms(args.pdb)
    phi = parse_indices(args.phi)
    psi = parse_indices(args.psi)

    expected_phi = find_expected_indices(atoms, EXPECTED_PHI_LABELS)
    expected_psi = find_expected_indices(atoms, EXPECTED_PSI_LABELS)

    phi_ok = report("phi", phi, labels_for_indices(atoms, phi), expected_phi, EXPECTED_PHI_LABELS)
    psi_ok = report("psi", psi, labels_for_indices(atoms, psi), expected_psi, EXPECTED_PSI_LABELS)

    compare_ok = True
    if args.compare_solution_pdb is not None:
        solution_atoms = read_pdb_atoms(args.compare_solution_pdb)
        compare_ok = compare_gas_solution_order(atoms, solution_atoms)
        if compare_ok:
            gas_phi_labels = labels_for_indices(atoms, phi)
            solution_phi_labels = labels_for_indices(solution_atoms, phi)
            gas_psi_labels = labels_for_indices(atoms, psi)
            solution_psi_labels = labels_for_indices(solution_atoms, psi)
            same_cv_labels = gas_phi_labels == solution_phi_labels and gas_psi_labels == solution_psi_labels
            print(f"gas/solution CV labels at same indices: {'OK' if same_cv_labels else 'FAIL'}")
            print(f"  gas phi labels:      {gas_phi_labels}")
            print(f"  solution phi labels: {solution_phi_labels}")
            print(f"  gas psi labels:      {gas_psi_labels}")
            print(f"  solution psi labels: {solution_psi_labels}")
            compare_ok = compare_ok and same_cv_labels

    if not (phi_ok and psi_ok and compare_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
