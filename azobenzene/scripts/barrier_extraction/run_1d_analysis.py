# run_1d_analysis.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .fes_io import parse_bias_log_1d
from .fes_reconstruct import reconstruct_fes_1d
from .plot_fes import EV_TO_KJMOL, EV_TO_KCALMOL

DATA_DIR = Path("azobenzene/outputs/full_runs")
OUT_ROOT = Path("azobenzene/analysis/barrier_extraction")


def _basins_1d(cv1, F):
    """cis basin lives in |cv1|<60°; trans basin lives in |cv1|>120°."""
    cis_window = (np.abs(cv1) < 60)
    trans_window = (np.abs(cv1) > 120)
    cis_i = int(np.where(cis_window)[0][np.argmin(F[cis_window])])
    trans_i = int(np.where(trans_window)[0][np.argmin(F[trans_window])])
    return cis_i, trans_i


def _barrier_1d_periodic(F, cis_i, trans_i):
    """Minimum-of-max along the periodic 1D circle: pick the arc
    (clockwise or counter-clockwise) with the lower barrier."""
    if cis_i == trans_i:
        raise ValueError(
            f"degenerate basin assignment: cis_i == trans_i == {cis_i}. "
            "Either both windows resolve to the same grid point (very narrow F) "
            "or _basins_1d failed; inspect the FES before trusting any barrier."
        )
    n = F.size
    if cis_i < trans_i:
        cw  = F[cis_i:trans_i + 1].max()
        ccw = max(F[trans_i:n].max(), F[:cis_i + 1].max())
    else:
        cw  = max(F[cis_i:n].max(), F[:trans_i + 1].max())
        ccw = F[trans_i:cis_i + 1].max()
    barrier_top = min(cw, ccw)
    return float(barrier_top - F[cis_i]), float(barrier_top - F[trans_i])


def process(txt_path: Path, grid_deg: float = 0.5, n_blocks: int = 5):
    run = parse_bias_log_1d(txt_path)
    cv1 = np.arange(-180.0, 180.0, grid_deg)
    F = reconstruct_fes_1d(run, cv1)
    cis_i, trans_i = _basins_1d(cv1, F)
    dG_c2t, dG_t2c = _barrier_1d_periodic(F, cis_i, trans_i)
    dG_rxn = float(F[cis_i] - F[trans_i])

    # block-averaged uncertainty (cumulative)
    n = run.height_eV.size
    lo = n // 2
    step = (n - lo) // n_blocks
    c2t_blk, t2c_blk, dG_blk = [], [], []
    for k in range(1, n_blocks + 1):
        nh = lo + k * step
        Fk = reconstruct_fes_1d(run, cv1, n_hills=nh)
        try:
            c_i, t_i = _basins_1d(cv1, Fk)
            a, b = _barrier_1d_periodic(Fk, c_i, t_i)
            c2t_blk.append(a); t2c_blk.append(b); dG_blk.append(float(Fk[c_i] - Fk[t_i]))
        except Exception:
            pass  # skip degenerate block

    sigma_c2t = float(np.std(c2t_blk, ddof=1)) if len(c2t_blk) > 1 else float("nan")
    sigma_t2c = float(np.std(t2c_blk, ddof=1)) if len(t2c_blk) > 1 else float("nan")
    sigma_rxn = float(np.std(dG_blk, ddof=1)) if len(dG_blk) > 1 else float("nan")

    out = {"tag": run.tag,
           "dG_c2t_eV": dG_c2t, "dG_c2t_sigma_eV": sigma_c2t,
           "dG_c2t_kJmol": dG_c2t * EV_TO_KJMOL,
           "dG_t2c_eV": dG_t2c, "dG_t2c_sigma_eV": sigma_t2c,
           "dG_t2c_kJmol": dG_t2c * EV_TO_KJMOL,
           "dG_rxn_eV": dG_rxn,  "dG_rxn_sigma_eV": sigma_rxn,
           "dG_rxn_kJmol": dG_rxn * EV_TO_KJMOL}

    # plot
    out_fig = OUT_ROOT / "figures" / run.tag
    out_fig.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cv1, F * EV_TO_KJMOL, "k-", lw=1.2)
    ax.axvline(cv1[cis_i], color="tab:blue", ls=":", label="cis min")
    ax.axvline(cv1[trans_i], color="tab:red", ls=":", label="trans min")
    ax.set_xlabel("CNNC dihedral (°)"); ax.set_ylabel("F (kJ/mol)")
    ax.set_title(f"1D MetaD FES — {run.tag}  ΔG‡(c→t)={dG_c2t*EV_TO_KJMOL:.1f} kJ/mol")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_fig / "fes_1d_native.png", dpi=160); plt.close(fig)
    return out


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for txt in sorted(DATA_DIR.glob("metad_azob_*_1d_*.txt")):
        print(f"[{txt.name}] processing …", flush=True)
        try:
            rows.append(process(txt))
        except Exception as e:
            rows.append({"tag": txt.name, "error": str(e)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "summary_table_1d.csv", index=False)
    print(f"Wrote {OUT_ROOT/'summary_table_1d.csv'}")


if __name__ == "__main__":
    main()
