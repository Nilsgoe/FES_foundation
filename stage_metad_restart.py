from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ase.io import Trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a decimated trajectory copy plus matching bias-file copy for MetaD restarts."
    )
    parser.add_argument("--input-traj", required=True)
    parser.add_argument("--input-bias", required=True)
    parser.add_argument("--output-traj", required=True)
    parser.add_argument("--output-bias", required=True)
    parser.add_argument("--stride", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_traj = Path(args.input_traj)
    input_bias = Path(args.input_bias)
    output_traj = Path(args.output_traj)
    output_bias = Path(args.output_bias)

    output_traj.parent.mkdir(parents=True, exist_ok=True)
    output_bias.parent.mkdir(parents=True, exist_ok=True)

    in_traj = Trajectory(str(input_traj))
    out_traj = Trajectory(str(output_traj), "w")
    last_index = len(in_traj) - 1
    written = 0
    for index, atoms in enumerate(in_traj):
        if index % args.stride == 0 or index == last_index:
            out_traj.write(atoms)
            written += 1
    out_traj.close()

    shutil.copy2(input_bias, output_bias)
    print(
        f"Staged restart copy: {input_traj} -> {output_traj} "
        f"with stride={args.stride}, frames_written={written}"
    )


if __name__ == "__main__":
    main()
