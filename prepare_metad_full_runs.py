from __future__ import annotations

import shutil
import subprocess
import sys
import argparse
from pathlib import Path


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
STAGER = ROOT / "stage_metad_restart.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=["all", "azobenzene", "malon", "f27329", "f27330"],
        default="all",
    )
    return parser.parse_args()


def iter_pairs(scope: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if scope in {"all", "azobenzene"}:
        for path in (ROOT / "azobenzene" / "outputs").glob("metad_*.traj"):
            pairs.append((path, path.with_suffix(".txt")))
    scoped_dirs = []
    if scope in {"all", "malon"}:
        scoped_dirs.append(ROOT / "malonaldehyd" / "ngoen_26984" / "outputs")
    if scope in {"all", "f27329"}:
        scoped_dirs.append(ROOT / "f-malonaldehyd" / "ngoen_27329" / "outputs")
    if scope in {"all", "f27330"}:
        scoped_dirs.append(ROOT / "f-malonaldehyd" / "ngoen_27330" / "outputs")
    for subdir in scoped_dirs:
        for path in subdir.glob("metad*.traj"):
            pairs.append((path, path.with_suffix(".txt")))
    return sorted(pairs)


def prepare_pair(traj: Path, txt: Path) -> None:
    full_runs = traj.parent / "full_runs"
    full_runs.mkdir(exist_ok=True)
    archived_traj = full_runs / traj.name
    archived_txt = full_runs / txt.name

    if traj.exists() and not archived_traj.exists():
        shutil.move(str(traj), str(archived_traj))
    if txt.exists() and not archived_txt.exists():
        shutil.move(str(txt), str(archived_txt))

    if not archived_traj.exists() or not archived_txt.exists():
        raise FileNotFoundError(f"Missing archived files for {traj}")

    if traj.exists():
        traj.unlink()
    if txt.exists():
        txt.unlink()

    subprocess.run(
        [
            sys.executable,
            str(STAGER),
            "--input-traj",
            str(archived_traj),
            "--input-bias",
            str(archived_txt),
            "--output-traj",
            str(traj),
            "--output-bias",
            str(txt),
            "--stride",
            "10",
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    pairs = iter_pairs(args.scope)
    for traj, txt in pairs:
        prepare_pair(traj, txt)
    print(f"prepared {len(pairs)} metad restart pairs for scope={args.scope}")


if __name__ == "__main__":
    main()
