#!/usr/bin/env python
import os
import csv
import numpy as np
import argparse
from jax import numpy as jnp
from ase.io import Trajectory

def calc_dihedral(positions):
    """
    Calculate the dihedral angle (in degrees) using positions of four atoms.
    Here we use indices [2, 11, 12, 13] from the positions array.
    
    Parameters:
        positions (np.ndarray): (N, 3) array of atomic positions.
    
    Returns:
        float: Dihedral angle in degrees.
    """
    # Extract the positions using np indexing. (Assumes positions has at least 14 atoms)
    p1, p2, p3, p4 = positions[np.array([2, 11, 12, 13])]
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    # Normalize normals (add a tiny value to avoid division by zero)
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    cos_phi = np.dot(n1, n2)
    sin_phi = np.dot(np.cross(n1, n2), b2)
    phi = np.arctan2(sin_phi, cos_phi)
    return np.degrees(phi).item()

def circular_mean(deg_angles):
    """
    Calculate the circular (angular) mean of a list of angles (in degrees)
    using the complex-exponential trick.
    """
    # Convert to radians
    rad_angles = np.radians(deg_angles)
    # Mean vector on the unit circle
    mean_vector = np.mean(np.exp(1j * rad_angles))
    return np.degrees(np.angle(mean_vector))

def process_trajectory(filename):
    """
    Load the trajectory (using ASE) from filename.
    For each frame, compute the dihedral angle with calc_dihedral.
    Returns:circular_mean
        tuple: (initial_dihedral, circular_mean_dihedral)
    """
    traj = Trajectory(filename)
    dihedrals = []
    for atoms in traj:
        pos = atoms.get_positions()
        angle = calc_dihedral(pos)
        dihedrals.append(angle)
    if not dihedrals:
        raise ValueError(f"No frames found in {filename}")
    initial_dihedral = dihedrals[0]
    mean_dihedral = circular_mean(dihedrals)
    return initial_dihedral, mean_dihedral

def main():
    parser = argparse.ArgumentParser(
        description="Process a trajectory file and compute its dihedral angles."
    )
    parser.add_argument("trajfile", type=str, help="Trajectory file to process")
    parser.add_argument(
        "--csv", type=str, default="dihedral_results.csv",
        help="CSV file to which results will be appended (default: dihedral_results.csv)"
    )
    args = parser.parse_args()
    
    try:
        init_val, mean_val = process_trajectory(args.trajfile)
        print(f"{args.trajfile}: initial={init_val:.2f} deg, mean={mean_val:.2f} deg")
    except Exception as e:
        print(f"Error processing {args.trajfile}: {e}")
        return
    
    # Append the result to the CSV file.
    file_exists = os.path.isfile(args.csv)
    with open(args.csv, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header if file did not exist previously
        if not file_exists:
            writer.writerow(["File", "Initial Dihedral (deg)", "Circular Mean Dihedral (deg)"])
        writer.writerow([args.trajfile, f"{init_val:.2f}", f"{mean_val:.2f}"])

if __name__ == "__main__":
    main()
