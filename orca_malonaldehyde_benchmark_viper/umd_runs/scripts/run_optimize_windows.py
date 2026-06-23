from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from ase.io import read

from run_umd_window import current_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize ORCA umbrella windows in nearest-neighbor order.")
    parser.add_argument("--windows-csv", type=Path, default=Path("windows.csv"))
    parser.add_argument("--xyz", type=Path, default=Path("malonaldehyde.xyz"))
    parser.add_argument("--optimized-root", type=Path, default=Path("optimized"))
    parser.add_argument("--output-root", type=Path, default=Path("optimization_work"))
    parser.add_argument("--start-shift", type=int)
    parser.add_argument("--end-shift", type=int)
    parser.add_argument("--order", choices=("nearest", "shift-ascending"), default="nearest")
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--opt-fmax", type=float, default=0.05)
    parser.add_argument("--opt-max-steps", type=int, default=200)
    parser.add_argument("--python", default="python")
    return parser.parse_args()


def read_windows(path: Path) -> list[dict[str, float | int]]:
    with path.open() as handle:
        return [
            {
                "window_id": int(row["window_id"]),
                "shift": int(row["shift"]),
                "center_A": float(row["center_A"]),
                "kappa_eV_A2": float(row["kappa_eV_A2"]),
            }
            for row in csv.DictReader(handle)
        ]


def main() -> None:
    args = parse_args()
    windows = read_windows(args.windows_csv)
    if args.start_shift is not None:
        windows = [window for window in windows if int(window["shift"]) >= args.start_shift]
    if args.end_shift is not None:
        windows = [window for window in windows if int(window["shift"]) <= args.end_shift]
    initial_cv = current_cv(read(args.xyz))
    if args.order == "shift-ascending":
        ordered = sorted(windows, key=lambda window: int(window["shift"]))
    else:
        ordered = sorted(windows, key=lambda window: abs(float(window["center_A"]) - initial_cv))

    args.optimized_root.mkdir(parents=True, exist_ok=True)
    with (args.optimized_root / "window_order.txt").open("w") as handle:
        handle.write(f"initial_cv_A={initial_cv}\n")
        handle.write(f"start_shift={args.start_shift}\n")
        handle.write(f"end_shift={args.end_shift}\n")
        handle.write(f"order={args.order}\n")
        for window in ordered:
            handle.write(
                f"{window['window_id']:03d} shift={window['shift']:+03d} "
                f"center_A={window['center_A']}\n"
            )

    for window in ordered:
        command = [
            args.python,
            "scripts/run_umd_window.py",
            "--window-id",
            str(window["window_id"]),
            "--windows-csv",
            str(args.windows_csv),
            "--xyz",
            str(args.xyz),
            "--optimized-root",
            str(args.optimized_root),
            "--output-root",
            str(args.output_root),
            "--use-nearest-optimized",
            "--optimize-only",
            "--opt-fmax",
            str(args.opt_fmax),
            "--opt-max-steps",
            str(args.opt_max_steps),
            "--steps",
            "0",
            "--cores",
            str(args.cores),
        ]
        print("Running", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
