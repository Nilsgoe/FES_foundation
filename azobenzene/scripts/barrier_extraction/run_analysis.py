# run_analysis.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .fes_io import parse_bias_log
from .fes_reconstruct import reconstruct_fes_2d, default_grid
from .convergence import fes_vs_time, basin_barrier_drift, converged
from .basins_mfep import (
    find_local_minima, nearest_grid, enumerate_pathways,
)
from .uncertainty import block_average, _pick_basin
from .projection_1d import project_to_cv1, cv2_wall_mask
from .plot_fes import (
    plot_2d_fes, plot_convergence_hills, plot_convergence_fes, plot_mfep_profile,
    EV_TO_KJMOL, EV_TO_KCALMOL,
)

DATA_DIR = Path("azobenzene/outputs/full_runs")
OUT_ROOT = Path("azobenzene/analysis/barrier_extraction")


def process_run(txt_path: Path, kT_eV: float, grid_deg: float = 1.0, n_blocks: int = 5):
    run = parse_bias_log(txt_path)
    out_data = OUT_ROOT / "data" / run.tag
    out_fig  = OUT_ROOT / "figures" / run.tag
    out_data.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    cv1, cv2 = default_grid(grid_deg, grid_deg)
    F = reconstruct_fes_2d(run, cv1, cv2)

    # Basins
    cis_min   = _pick_basin(F, cv1, cv2, target=(0.0, 120.0))
    trans_min = _pick_basin(F, cv1, cv2, target=(180.0, 120.0))
    F_cis   = float(F[cis_min])
    F_trans = float(F[trans_min])

    # Pathways
    paths = enumerate_pathways(F, cv1, cv2, cis_min, trans_min)

    # Convergence
    snaps = fes_vs_time(run, cv1, cv2)
    basin_mask = np.zeros_like(F, dtype=bool)
    for name, (i, j) in {"cis": cis_min, "trans": trans_min}.items():
        di = max(3, int(round(20.0 / grid_deg)))
        ii_lo, ii_hi = max(0, i - di), min(F.shape[0], i + di + 1)
        jj_lo, jj_hi = max(0, j - di), min(F.shape[1], j + di + 1)
        basin_mask[ii_lo:ii_hi, jj_lo:jj_hi] = True
    barrier_mask = np.zeros_like(F, dtype=bool)
    for name, info in paths.items():
        n2 = F.shape[1]
        ii = info["path_idx"] // n2
        jj = info["path_idx"] %  n2
        barrier_mask[ii, jj] = True
    drift = basin_barrier_drift(snaps, basin_mask, barrier_mask)
    conv_ok = converged(drift, threshold_eV=0.05)

    # Block-averaged uncertainty.
    # Per Addendum D: cumulative-block std under-estimates because adjacent blocks share
    # data. The block-to-block (consecutive-difference) std captures residual drift and
    # is a more conservative estimator. We report BOTH and use sigma = max(cum, b2b)
    # as the headline uncertainty.
    blocks, F_blocks = block_average(run, cv1, cv2, n_blocks=n_blocks)
    blk_arr = np.array([[b.barrier_rot_cis_to_trans_eV,
                         b.barrier_rot_trans_to_cis_eV,
                         b.barrier_inv_cis_to_trans_eV,
                         b.barrier_inv_trans_to_cis_eV,
                         b.dG_rxn_eV] for b in blocks])
    means = blk_arr.mean(axis=0)
    stds_cum = blk_arr.std(axis=0, ddof=1)
    if blk_arr.shape[0] >= 3:
        stds_b2b = np.diff(blk_arr, axis=0).std(axis=0, ddof=1)
    else:
        stds_b2b = stds_cum
    stds = np.maximum(stds_cum, stds_b2b)   # conservative

    # 1D projection
    F_1d = project_to_cv1(F, cv1, cv2, kT_eV=kT_eV,
                          cv2_mask=cv2_wall_mask(cv2, margin_deg=5.0))

    # Save artefacts
    np.savez_compressed(out_data / "fes_2d.npz",
                        cv1=cv1, cv2=cv2, F_eV=F,
                        cis_min_ij=np.array(cis_min), trans_min_ij=np.array(trans_min))
    np.savez_compressed(out_data / "fes_blocks.npz", F_blocks=F_blocks)
    pd.DataFrame({"cv1_deg": cv1,
                  "F_1d_eV": F_1d,
                  "F_1d_kJmol": F_1d * EV_TO_KJMOL}).to_csv(
        out_data / "fes_1d.csv", index=False)
    for name, info in paths.items():
        n2 = F.shape[1]
        ii = info["path_idx"] // n2; jj = info["path_idx"] % n2
        pd.DataFrame({"step": np.arange(ii.size),
                      "cv1_deg": cv1[ii], "cv2_deg": cv2[jj],
                      "F_eV": info["F_path"],
                      "F_kJmol": info["F_path"] * EV_TO_KJMOL}
                    ).to_csv(out_data / f"mfep_{name}.csv", index=False)

    # Figures
    plot_2d_fes(F, cv1, cv2, out_fig / "fes_2d.png",
                title=f"{run.tag}  (kT={kT_eV*1000:.1f} meV; converged={conv_ok})",
                basins={"cis": cis_min, "trans": trans_min}, paths=paths)
    plot_convergence_hills(run.time_fs, run.height_eV,
                           out_fig / "convergence_hills.png",
                           title=f"WT hill heights — {run.tag}")
    plot_convergence_fes(snaps, cv1, cv2, out_fig / "convergence_fes.png",
                         title=f"FES min-over-CV2 vs hill count — {run.tag}")
    plot_mfep_profile(paths, F, out_fig / "mfep_profile.png",
                      title=f"MFEP profiles — {run.tag}")

    return {
        "tag": run.tag,
        "n_hills": int(run.height_eV.size),
        "sim_time_ps": float(run.time_fs[-1] / 1000.0),
        "converged_drift_eV": float(drift[-1]["max_abs_barrier_eV"]),
        "converged": bool(conv_ok),
        "F_cis_eV": F_cis,
        "F_trans_eV": F_trans,
        "barriers_eV": {
            "rot_cis_to_trans_mean":   float(means[0]),
            "rot_cis_to_trans_std":    float(stds[0]),
            "rot_trans_to_cis_mean":   float(means[1]),
            "rot_trans_to_cis_std":    float(stds[1]),
            "inv_cis_to_trans_mean":   float(means[2]),
            "inv_cis_to_trans_std":    float(stds[2]),
            "inv_trans_to_cis_mean":   float(means[3]),
            "inv_trans_to_cis_std":    float(stds[3]),
            "dG_rxn_mean":             float(means[4]),
            "dG_rxn_std":              float(stds[4]),
        },
        "stds_cumulative_eV": [float(x) for x in stds_cum],
        "stds_block_to_block_eV": [float(x) for x in stds_b2b],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--temperature_K", type=float, default=333.0,
                   help="Production T for kT in the 1D projection. "
                        "333 K matches run_metad.py:177,182.")
    p.add_argument("--grid_deg", type=float, default=1.0)
    p.add_argument("--n_blocks", type=int, default=5)
    p.add_argument("--pattern", default="metad_azob_*_2d_*.txt")
    args = p.parse_args()

    kT_eV = 8.617333e-5 * args.temperature_K   # k_B in eV/K

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for txt in sorted(DATA_DIR.glob(args.pattern)):
        print(f"[{txt.name}] processing …", flush=True)
        try:
            summaries.append(process_run(txt, kT_eV=kT_eV,
                                         grid_deg=args.grid_deg, n_blocks=args.n_blocks))
        except Exception as e:
            print(f"  ERROR: {e}")
            summaries.append({"tag": txt.name, "error": str(e)})

    rows = []
    for s in summaries:
        if "error" in s:
            rows.append({"tag": s["tag"], "error": s["error"]}); continue
        b = s["barriers_eV"]
        rows.append({
            "tag": s["tag"],
            "n_hills": s["n_hills"],
            "sim_time_ps": s["sim_time_ps"],
            "converged": s["converged"],
            "drift_eV": s["converged_drift_eV"],
            **{f"{k}_eV": v for k, v in b.items()},
            **{f"{k}_kJmol": v * 96.485 for k, v in b.items()},
            **{f"{k}_kcalmol": v * 23.061 for k, v in b.items()},
        })
    pd.DataFrame(rows).to_csv(OUT_ROOT / "summary_table.csv", index=False)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"Wrote {OUT_ROOT}/summary_table.csv and summary.json")


if __name__ == "__main__":
    main()
