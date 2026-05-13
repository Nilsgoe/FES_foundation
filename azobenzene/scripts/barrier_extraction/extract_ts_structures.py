# extract_ts_structures.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from ase.io import iread, write
from .fes_io import parse_bias_log

OUT = Path("azobenzene/analysis/barrier_extraction/ts_structures")

# CV index sets (from run_metad.py SYSTEM_SPECS)
INDICES = {
    "cis":   {"dihedral": (1, 6, 7, 8),  "angle": (1, 6, 7)},
    "trans": {"dihedral": (2, 11, 12, 13), "angle": (2, 11, 12)},
}


def _cv_from_atoms(atoms, dihedral_idx, angle_idx):
    pos = atoms.get_positions()
    # dihedral
    p1, p2, p3, p4 = pos[list(dihedral_idx)]
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2); y = np.dot(m1, n2)
    dih = np.degrees(np.arctan2(y, x))
    # angle
    a, b, c = pos[list(angle_idx)]
    ba = a - b; bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return dih, ang


def extract_for_run(txt_path: Path, ts_points: dict[str, tuple[float, float]],
                    stride: int = 200) -> dict[str, float]:
    """For each named saddle (cv1°, cv2°), find the trajectory frame
    minimizing (Δcv1²+Δcv2²) and write it as .xyz. Returns the best-match
    score per saddle (squared distance in deg²; 0 = exact match)."""
    traj_path = txt_path.with_suffix(".traj")
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)
    seed = "cis" if "_cis_" in txt_path.name else "trans"
    idx = INDICES[seed]

    best = {name: (np.inf, None) for name in ts_points}
    for atoms in iread(str(traj_path), index=f"::{stride}"):
        try:
            cv1, cv2 = _cv_from_atoms(atoms, idx["dihedral"], idx["angle"])
        except Exception:
            continue
        for name, (t1, t2) in ts_points.items():
            d1 = ((cv1 - t1 + 180) % 360) - 180   # periodic CV1
            score = d1 * d1 + (cv2 - t2) ** 2
            if score < best[name][0]:
                best[name] = (score, atoms.copy())

    run_tag = parse_bias_log(txt_path).tag
    out = OUT / run_tag
    out.mkdir(parents=True, exist_ok=True)
    scores = {}
    for name, (score, atoms) in best.items():
        if atoms is None:
            continue
        write(out / f"ts_{name}.xyz", atoms)
        scores[name] = float(score)
    return scores


def saddle_from_run(tag: str) -> dict[str, tuple[float, float]]:
    """Read the per-run mfep_*.csv files and locate the saddle (path point
    with max F) for rotation and inversion pathways. Returns dict of
    name -> (cv1°, cv2°)."""
    import pandas as pd
    data_dir = Path("azobenzene/analysis/barrier_extraction/data") / tag
    saddles = {}
    for name in ("rotation", "inversion"):
        csv = data_dir / f"mfep_{name}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        k = int(df["F_eV"].idxmax())
        saddles[name] = (float(df.loc[k, "cv1_deg"]), float(df.loc[k, "cv2_deg"]))
    return saddles


def main():
    DATA_DIR = Path("azobenzene/outputs/full_runs")
    OUT.mkdir(parents=True, exist_ok=True)
    all_scores = {}
    for txt in sorted(DATA_DIR.glob("metad_azob_*_2d_*.txt")):
        run = parse_bias_log(txt)
        saddles = saddle_from_run(run.tag)
        if not saddles:
            print(f"[{run.tag}] no MFEP CSV found — skipping")
            continue
        print(f"[{run.tag}] saddles: " + ", ".join(
            f"{n}=(cv1={s[0]:.1f}°, cv2={s[1]:.1f}°)" for n, s in saddles.items()))
        scores = extract_for_run(txt, saddles, stride=200)
        all_scores[run.tag] = scores
        for name, sc in scores.items():
            print(f"  {name}: match score = {sc:.2f} deg² (lower is better)")
    print("Done")
    return all_scores


if __name__ == "__main__":
    main()
