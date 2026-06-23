from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


WINDOW_RE = re.compile(r"window_(\d+)_shift_([+-]\d+)(?:_cv_energy\.csv)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed-duration ORCA DFT UMD analysis snapshot.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ps", type=float, required=True)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument(
        "--mirror-cv",
        action="store_true",
        help="Negate the CV and umbrella centers, preserving the underlying configurations and energies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_count = round(args.ps * 1000.0 / args.timestep_fs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.windows_csv.open(newline="") as src, (args.output_dir / "windows.csv").open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if args.mirror_cv:
                row["center_A"] = str(-float(row["center_A"]))
            writer.writerow(row)

    written = 0
    sources = list(args.source_dir.glob("window_*/cv_energy.csv"))
    sources.extend(args.source_dir.glob("window_*_cv_energy.csv"))
    for source in sorted(sources):
        source_label = source.parent.name if source.name == "cv_energy.csv" else source.name
        match = WINDOW_RE.fullmatch(source_label)
        if not match:
            continue
        window_id, shift = int(match.group(1)), int(match.group(2))
        target = args.output_dir / f"window_{window_id:03d}_shift_{shift:+03d}_cv_energy.csv"
        with source.open(newline="") as src, target.open("w", newline="") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)
            header = next(reader)
            writer.writerow(header)
            cv_index = header.index("cv_A")
            rows = 0
            for row in reader:
                if rows >= sample_count:
                    break
                if args.mirror_cv:
                    row[cv_index] = str(-float(row[cv_index]))
                writer.writerow(row)
                rows += 1
        if rows < sample_count:
            raise RuntimeError(f"{source} has only {rows} samples; {sample_count} required for {args.ps:g} ps")
        written += 1

    if written != 41:
        raise RuntimeError(f"Expected 41 windows, wrote {written}")
    print(f"Wrote {written} windows x {sample_count} samples ({args.ps:g} ps) to {args.output_dir}")


if __name__ == "__main__":
    main()
