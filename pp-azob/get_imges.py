#!/usr/bin/env python3
import argparse
from ase.io import read, write

def main():
    parser = argparse.ArgumentParser(
        description="Extract two specific frames from a large trajectory and save them as SVG images."
    )
    parser.add_argument("traj_file", help="Path to the trajectory file (e.g. traj.traj)")
    parser.add_argument("indices", type=int, nargs=2,
                        help="Two indices of the structures to extract (e.g. 10 50)")
    args = parser.parse_args()

    for idx in args.indices:
        try:
            # Read a single frame (structure) from the trajectory.
            atoms = read(args.traj_file, index=idx)
        except Exception as e:
            print(f"Error reading index {idx} from {args.traj_file}: {e}")
            continue

        # Define an output filename for the SVG image.
        svg_filename = f"structure_{idx}.png"
        # Write the structure to an SVG file.
        write(svg_filename, atoms, format="png",scale=200)
        print(f"Structure at index {idx} written to {svg_filename}")


if __name__ == "__main__":
    main()
