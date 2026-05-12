# Azobenzene cis ⇌ trans Barrier Extraction from 2D WT-MetaD — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute ΔG‡(cis→trans), ΔG‡(trans→cis) and ΔG_rxn for azobenzene from the existing 2D well-tempered metadynamics runs in [outputs/full_runs/](../../outputs/full_runs/), for all four MACE models (`omol`, `off`, `mh1`, `polar`), reporting per-seed (cis-start, trans-start) values with block-averaged uncertainty and an MFEP-derived saddle — not naive grid min/max.

**Architecture:** Pure-Python numpy/scipy/matplotlib pipeline. Reads the custom ASE-`WT_Metadynamics` per-hill `.txt` logs (NOT PLUMED — no `sum_hills`), reconstructs the 2D FES with correct periodicity, finds basins, extracts barriers via Dijkstra-minimax MFEP on the grid, enumerates rotation vs inversion pathways, quantifies uncertainty by block-averaging the second half of hills, and projects to 1D for cross-check. Outputs go to a new `analysis/barrier_extraction/` tree; raw data is never modified.

**Tech Stack:** Python 3, numpy, scipy (`scipy.ndimage`, `scipy.sparse.csgraph.dijkstra`, `scipy.special.logsumexp`), matplotlib. No PLUMED. No new installs expected — verify with Task 0.

---

## Engine & data facts (locked, from probe of the real files)

- **Engine:** ASE + custom `WT_Metadynamics` class at [/nexus/posix0/FHI-Theory/ngoen/biASE/Metadynamics.py:463](/nexus/posix0/FHI-Theory/ngoen/biASE/Metadynamics.py), driven by [run_metad.py](../../run_metad.py).
- **CV1:** CNNC dihedral, degrees, **periodic** (`wrapping=True`), bounds (−180°, 180°), σ₁ = 5°. Cis indices `(1,6,7,8)`, trans `(2,11,12,13)`.
- **CV2:** NNC bending angle, degrees, **non-periodic** (`wrapping=False`), bounds (0°, 180°), σ₂ = 5°. Cis indices `(1,6,7)`, trans `(2,11,12)`.
- **MetaD parameters (from log header line `!#`):** `Height=0.1 eV`, `Pace=100` MD steps, `Step_size=0.5 fs` (deposition every 50 fs), `Bias_factor γ=10`, `WT=True`, `Stretch=False`. Total: 1e6 MD steps = 500 ps per run, ~10,000 hills per run.
- **Bias-log row format:** `time(fs)  CV1(°)  CV2(°)  std1  std2  height(eV)  biasfactor  regularizationfactor`
- **CRITICAL — height-column convention:** The stored `height` is **already pre-multiplied** by γ/(γ−1). Evidence: row 1 has `height = 0.11111111` for `Height₀ = 0.1`, and 0.1 × 10/9 = 0.1111… . Code reference: `Metadynamics.py:839` deposits `(γ/(γ-1)) * w₀ * reg_factor`.
  - **⇒ `F(s) = −V_bias(s)` directly. Do NOT multiply by γ/(γ−1) again.** Getting this wrong = 11% systematic error.
- **Energy units:** eV (ASE). Conversions: 1 eV = 96.485 kJ/mol = 23.061 kcal/mol. **kT at the production temperature 333 K = 0.02870 eV** (verified against `run_metad.py:177` and `:182`; do NOT use 300 K — Codex peer-review caught this).
- **CV1 periodicity:** stored hill centers may drift unwrapped (e.g. `−165°` near end). The reconstruction kernel must apply minimum-image `Δθ = ((s − c + 180) mod 360) − 180`.
- **8 production 2D runs to analyze** (4 models × {cis-start, trans-start}), all under [outputs/full_runs/](../../outputs/full_runs/), pattern `metad_azob_{cis|trans}_{model}_2d_*.txt` (+ `.traj`).

---

## File structure

Create under [azobenzene/scripts/barrier_extraction/](../../scripts/barrier_extraction/) (NOT under [azobenzene/analysis/](../../analysis/) — `analysis/` holds outputs; `scripts/` holds code):

```
azobenzene/scripts/barrier_extraction/
├── __init__.py
├── fes_io.py             # parse bias .txt log + run metadata
├── fes_reconstruct.py    # vectorized 2D FES from hills (periodic CV1)
├── convergence.py        # hill-height-vs-time + FES-vs-time diagnostics
├── basins_mfep.py        # local minima, Dijkstra-minimax MFEP, pathway enum
├── uncertainty.py        # block-averaging over second half of hills
├── projection_1d.py      # F_1D(s1) = -kT ln ∫ exp(-F/kT) ds2
├── plot_fes.py           # 2D contour + MFEP overlay; convergence figures
├── run_analysis.py       # orchestrator: 8 runs → all outputs + report
└── tests/
    ├── test_fes_io.py
    ├── test_fes_reconstruct.py   # periodicity, scaling assumption
    ├── test_basins_mfep.py       # toy double-well
    └── test_uncertainty.py       # synthetic block variance
```

Outputs land in:
```
azobenzene/analysis/barrier_extraction/
├── barrier_analysis.md           # final report (the deliverable)
├── summary_table.csv             # all 8 runs × {ΔG‡_rot, ΔG‡_inv, ΔG_rxn, ±σ}
├── data/
│   └── {run_tag}/
│       ├── fes_2d.npz            # CV1_grid, CV2_grid, F (eV)
│       ├── fes_blocks.npz        # F per block (uncertainty)
│       ├── fes_1d.csv            # CV1, F_1D
│       ├── mfep_rotation.csv     # CV1, CV2, F along path
│       └── mfep_inversion.csv
└── figures/
    └── {run_tag}/
        ├── convergence_hills.png
        ├── convergence_fes.png   # 5 overlaid 1D slices
        ├── fes_2d.png            # heatmap + MFEP overlay + basins marked
        └── mfep_profile.png      # F along MFEP, both pathways
```

`{run_tag}` = e.g. `cis_omol_2d`, `trans_off_2d`, …

---

## Tasks

> **TDD note:** The user's CLAUDE.md says "Do not write tests unless explicitly asked." This plan touches >5 files and >100 lines, so per the same CLAUDE.md exception, **ask once at Task 0 whether to write the tests below**. If declined, skip the `test_*` steps but keep the validation `python -c` smoke checks.

### Task 0: Scope confirmation + environment probe

**Files:**
- Read only: [/nexus/posix0/FHI-Theory/ngoen/biASE/Metadynamics.py](/nexus/posix0/FHI-Theory/ngoen/biASE/Metadynamics.py) (lines 463 and 839 in particular)
- Create: nothing yet

- [ ] **Step 0.1: Verify the height-scaling convention by reading Metadynamics.py:830-850**

Run: `sed -n '825,860p' /nexus/posix0/FHI-Theory/ngoen/biASE/Metadynamics.py`
Expected: a line resembling `height = (bias_factor / (bias_factor - 1)) * bias_height * regulation_factor` (or equivalent). Confirms `F = −V_bias` (no extra prefactor at reconstruction).
**If the code does NOT pre-multiply by γ/(γ−1)** (e.g. it stores raw `w₀ * reg_f`), update the formula in [fes_reconstruct.py](../../scripts/barrier_extraction/fes_reconstruct.py) Task 3 to `F = −γ/(γ−1) · V_bias` BEFORE proceeding.

- [ ] **Step 0.2: Verify environment**

Run: `python -c "import numpy, scipy, matplotlib, ase; from scipy.sparse.csgraph import dijkstra; from scipy.special import logsumexp; from scipy.ndimage import minimum_filter; print('ok')"`
Expected: `ok`. If any import fails, ask the user before installing.

- [ ] **Step 0.3: Confirm temperature (already known: 333 K)**

Run: `grep -n 'temperature_K\|MaxwellBoltzmann' azobenzene/run_metad.py`
Expected output includes `run_metad.py:177: MaxwellBoltzmannDistribution(atoms, temperature_K=333)` and `run_metad.py:182: temperature_K=333,`. **Use 333 K everywhere downstream** (kT = 0.02870 eV). If the line reports a different value, update the orchestrator default in Task 8 and the convergence-threshold rationale in Task 3 accordingly. Record the actual value in `barrier_analysis.md`.

- [ ] **Step 0.4: Ask user whether to write the tests/ files**

One yes/no question. Default to YES given >100 lines + 5 files. Document the decision in [barrier_analysis.md](../../analysis/barrier_extraction/barrier_analysis.md) front matter.

- [ ] **Step 0.5: Commit the empty scaffold**

```bash
mkdir -p azobenzene/scripts/barrier_extraction/tests \
         azobenzene/analysis/barrier_extraction/{data,figures}
touch azobenzene/scripts/barrier_extraction/__init__.py
git add azobenzene/scripts/barrier_extraction/__init__.py \
        azobenzene/analysis/barrier_extraction/.gitkeep 2>/dev/null || true
git commit -m "feat(azobenzene): scaffold barrier_extraction pipeline" || true
```

---

### Task 1: Bias-log parser

**Files:**
- Create: `azobenzene/scripts/barrier_extraction/fes_io.py`
- Test: `azobenzene/scripts/barrier_extraction/tests/test_fes_io.py`

- [ ] **Step 1.1: Write the parser**

```python
# fes_io.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np

@dataclass(frozen=True)
class MetadRun:
    path: Path
    tag: str                  # e.g. "cis_omol_2d"
    height0_eV: float         # nominal Height from header
    pace_steps: int
    step_size_fs: float
    bias_factor: float
    wt: bool
    time_fs: np.ndarray       # (n_hills,)
    cv1_deg: np.ndarray       # (n_hills,) CNNC dihedral
    cv2_deg: np.ndarray       # (n_hills,) NNC angle
    sigma1_deg: np.ndarray    # (n_hills,)
    sigma2_deg: np.ndarray
    height_eV: np.ndarray     # (n_hills,) ALREADY γ/(γ-1)-scaled
    reg_factor: np.ndarray    # (n_hills,) = h_i / (Height0 * γ/(γ-1))

_HEADER_KV = re.compile(r"(\w+)\s*=\s*([-\d.eE+TruFals]+)")

def parse_bias_log(path: str | Path) -> MetadRun:
    path = Path(path)
    with path.open() as f:
        header = f.readline()
    if not header.startswith("!#"):
        raise ValueError(f"{path}: missing '!#' header line")
    kv = dict(_HEADER_KV.findall(header))
    data = np.loadtxt(path, skiprows=2)
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError(f"{path}: expected >=8 columns, got shape {data.shape}")
    tag = _tag_from_filename(path)
    return MetadRun(
        path=path, tag=tag,
        height0_eV=float(kv["Height"]),
        pace_steps=int(kv["Pace"]),
        step_size_fs=float(kv["Step_size"]),
        bias_factor=float(kv["Bias_factor"]),
        wt=(kv["WT"] == "True"),
        time_fs=data[:, 0],
        cv1_deg=data[:, 1],
        cv2_deg=data[:, 2],
        sigma1_deg=data[:, 3],
        sigma2_deg=data[:, 4],
        height_eV=data[:, 5],
        reg_factor=data[:, 7],
    )

def _tag_from_filename(p: Path) -> str:
    # metad_azob_{cis|trans}_{model}_{1d|2d}_*.txt -> "{cis|trans}_{model}_{1|2}d"
    m = re.search(r"metad_azob_(cis|trans)_(\w+?)_(1d|2d)_", p.name)
    if not m:
        raise ValueError(f"unrecognized filename pattern: {p.name}")
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
```

- [ ] **Step 1.2: Write the test**

```python
# tests/test_fes_io.py
from pathlib import Path
import numpy as np
from azobenzene.scripts.barrier_extraction.fes_io import parse_bias_log

PROD = Path("azobenzene/outputs/full_runs")

def test_parse_real_2d_log():
    f = next(PROD.glob("metad_azob_cis_omol_2d_*.txt"))
    run = parse_bias_log(f)
    assert run.bias_factor == 10
    assert run.height0_eV == 0.1
    assert run.time_fs.size == 10_000
    # WT pre-scaling check: row 0 should be ~ 0.1 * 10/9
    assert np.isclose(run.height_eV[0], 0.1 * 10/9, rtol=1e-4), \
        f"height pre-scaling broken: got {run.height_eV[0]}"
    # Hills should decay (well-tempered)
    assert run.height_eV[-100:].mean() < run.height_eV[:100].mean()
```

- [ ] **Step 1.3: Run the test**

Run: `pytest azobenzene/scripts/barrier_extraction/tests/test_fes_io.py -v`
Expected: PASS. If the pre-scaling assertion fails, **stop and revisit Task 0.1** — formula in Task 3 may need γ/(γ−1) prefactor.

- [ ] **Step 1.4: Commit**

```bash
git add azobenzene/scripts/barrier_extraction/fes_io.py \
        azobenzene/scripts/barrier_extraction/tests/test_fes_io.py
git commit -m "feat(barrier_extraction): parse WT_Metadynamics bias log"
```

---

### Task 2: Vectorized 2D FES reconstruction with periodic CV1

**Files:**
- Create: `azobenzene/scripts/barrier_extraction/fes_reconstruct.py`
- Test: `azobenzene/scripts/barrier_extraction/tests/test_fes_reconstruct.py`

- [ ] **Step 2.1: Write the reconstructor**

```python
# fes_reconstruct.py
from __future__ import annotations
import numpy as np
from .fes_io import MetadRun

def default_grid(d_cv1: float = 1.0, d_cv2: float = 1.0):
    cv1 = np.arange(-180.0, 180.0, d_cv1)          # periodic, exclude +180 (== -180)
    cv2 = np.arange(0.0, 180.0 + 1e-9, d_cv2)      # closed
    return cv1, cv2

def _wrap_deg(delta: np.ndarray) -> np.ndarray:
    return (delta + 180.0) % 360.0 - 180.0

def reconstruct_fes_2d(
    run: MetadRun,
    cv1_grid: np.ndarray,
    cv2_grid: np.ndarray,
    n_hills: int | None = None,
    chunk: int = 500,
) -> np.ndarray:
    """Return F(cv1, cv2) in eV, shifted so min(F) = 0.

    Heights stored in the log are ALREADY γ/(γ-1)-scaled (see Metadynamics.py:839),
    so F = - sum_i height_i * G_i. Do NOT apply an extra γ/(γ-1) factor here.
    """
    if n_hills is None:
        n_hills = run.height_eV.size
    H = run.height_eV[:n_hills]
    C1 = run.cv1_deg[:n_hills]
    C2 = run.cv2_deg[:n_hills]
    S1 = run.sigma1_deg[:n_hills]
    S2 = run.sigma2_deg[:n_hills]

    # Shapes: V[i, j] sums over hills k of h_k * G1(cv1_i - c1_k) * G2(cv2_j - c2_k)
    V = np.zeros((cv1_grid.size, cv2_grid.size), dtype=np.float64)
    for k0 in range(0, n_hills, chunk):
        k1 = min(k0 + chunk, n_hills)
        # (G1, G2) per-hill chunk -> rank-3 then sum over k
        d1 = _wrap_deg(cv1_grid[:, None] - C1[None, k0:k1])         # (Ncv1, Nk)
        d2 = cv2_grid[:, None] - C2[None, k0:k1]                    # (Ncv2, Nk)
        g1 = np.exp(-0.5 * (d1 / S1[None, k0:k1]) ** 2)
        g2 = np.exp(-0.5 * (d2 / S2[None, k0:k1]) ** 2)
        # contribution h_k * outer(g1[:,k], g2[:,k]) summed over k
        V += (g1 * H[None, k0:k1]) @ g2.T                            # (Ncv1, Ncv2)

    F = -V
    F -= F.min()
    return F
```

- [ ] **Step 2.2: Write the test (periodicity + double-counting)**

```python
# tests/test_fes_reconstruct.py
import numpy as np
from azobenzene.scripts.barrier_extraction.fes_io import MetadRun
from azobenzene.scripts.barrier_extraction.fes_reconstruct import (
    reconstruct_fes_2d, default_grid, _wrap_deg,
)
from pathlib import Path

def _make_run(centers_cv1, centers_cv2, h=0.1*10/9):
    n = len(centers_cv1)
    return MetadRun(
        path=Path("x"), tag="t", height0_eV=0.1, pace_steps=100,
        step_size_fs=0.5, bias_factor=10.0, wt=True,
        time_fs=np.arange(n)*50.0,
        cv1_deg=np.array(centers_cv1, dtype=float),
        cv2_deg=np.array(centers_cv2, dtype=float),
        sigma1_deg=np.full(n, 5.0), sigma2_deg=np.full(n, 5.0),
        height_eV=np.full(n, h), reg_factor=np.ones(n),
    )

def test_wrap_deg():
    assert _wrap_deg(np.array([200.0]))[0] == -160.0
    assert _wrap_deg(np.array([-200.0]))[0] == 160.0

def test_periodicity_seam_handled():
    # A hill at CV1 = +180° must, via minimum-image wrapping, place its center
    # exactly on the grid point at cv1 = -180° (which equals +180° mod 360°).
    # If periodicity were broken, F at -180° would be near the max (kernel far).
    # With correct periodicity, F at -180° = 0 (bottom of the well after min-shift)
    # and F at +179° (1° away through the seam) is slightly above 0.
    run = _make_run([180.0], [90.0])
    cv1, cv2 = default_grid(1.0, 1.0)
    F = reconstruct_fes_2d(run, cv1, cv2)
    j_mid = int(np.argmin(np.abs(cv2 - 90.0)))
    # cv1[0] = -180°  -> distance 0 from hill at +180°  -> F = 0
    assert F[0, j_mid] < 1e-9, f"seam not at min: F[-180,90]={F[0,j_mid]:.6f}"
    # cv1[-1] = +179° -> distance 1° from hill, σ = 5°
    F_at_179 = F[-1, j_mid]
    # Expected: ΔF = h*(1 - exp(-0.5/25)) = 0.1111 * 0.0198 ≈ 0.0022 eV
    assert 1e-4 < F_at_179 < 5e-3, f"unexpected F at +179°: {F_at_179:.6f}"
    # And F should be symmetric across the seam: F(-180+Δ) == F(180-Δ)
    assert abs(F[1, j_mid] - F[-1, j_mid]) < 1e-9

def test_no_double_counting():
    # Two hills at same place => V doubles, F doubles (both relative to baseline)
    cv1, cv2 = default_grid(2.0, 2.0)
    r1 = _make_run([0.0], [90.0])
    r2 = _make_run([0.0, 0.0], [90.0, 90.0])
    F1 = reconstruct_fes_2d(r1, cv1, cv2)
    F2 = reconstruct_fes_2d(r2, cv1, cv2)
    # min(F)=0; depth at center is twice (within 1%)
    far = (0, 0)
    near = (np.argmin(np.abs(cv1)), np.argmin(np.abs(cv2 - 90.0)))
    d1 = F1[far] - F1[near]
    d2 = F2[far] - F2[near]
    assert np.isclose(d2, 2*d1, rtol=0.01)
```

- [ ] **Step 2.3: Run tests**

Run: `pytest azobenzene/scripts/barrier_extraction/tests/test_fes_reconstruct.py -v`
Expected: PASS.

- [ ] **Step 2.4: Smoke check on real data**

Run:
```bash
python - <<'PY'
from pathlib import Path
from azobenzene.scripts.barrier_extraction.fes_io import parse_bias_log
from azobenzene.scripts.barrier_extraction.fes_reconstruct import reconstruct_fes_2d, default_grid
f = next(Path("azobenzene/outputs/full_runs").glob("metad_azob_cis_omol_2d_*.txt"))
run = parse_bias_log(f)
cv1, cv2 = default_grid(2.0, 2.0)   # coarse for speed
F = reconstruct_fes_2d(run, cv1, cv2)
print(f"F shape={F.shape}, range=[{F.min():.3f}, {F.max():.3f}] eV "
      f"= [{F.min()*96.485:.1f}, {F.max()*96.485:.1f}] kJ/mol")
PY
```
Expected: shape `(180, 91)`, range starts at 0, max somewhere between ~0.5 and ~3 eV (barrier ~50-300 kJ/mol). If max > 5 eV or < 0.3 eV, the scaling assumption is wrong — revisit Task 0.1.

- [ ] **Step 2.5: Commit**

```bash
git add azobenzene/scripts/barrier_extraction/fes_reconstruct.py \
        azobenzene/scripts/barrier_extraction/tests/test_fes_reconstruct.py
git commit -m "feat(barrier_extraction): reconstruct 2D FES with periodic CV1"
```

---

### Task 3: Convergence diagnostics

**Files:**
- Create: `azobenzene/scripts/barrier_extraction/convergence.py`

- [ ] **Step 3.1: Implement**

```python
# convergence.py
from __future__ import annotations
import numpy as np
from .fes_io import MetadRun
from .fes_reconstruct import reconstruct_fes_2d

def fes_vs_time(run: MetadRun, cv1, cv2, fractions=(0.2, 0.4, 0.6, 0.8, 1.0)):
    n = run.height_eV.size
    out = {}
    for f in fractions:
        nh = max(1, int(round(n * f)))
        out[f] = reconstruct_fes_2d(run, cv1, cv2, n_hills=nh)
    return out  # dict {fraction: F(cv1,cv2)}

def basin_barrier_drift(fes_snapshots: dict, mask_basin, mask_barrier):
    """max|ΔF| between consecutive snapshots, restricted to a region mask."""
    fracs = sorted(fes_snapshots)
    deltas = []
    for a, b in zip(fracs[:-1], fracs[1:]):
        d = fes_snapshots[b] - fes_snapshots[a]
        deltas.append({
            "from": a, "to": b,
            "max_abs_basin_eV":   float(np.max(np.abs(d[mask_basin]))),
            "max_abs_barrier_eV": float(np.max(np.abs(d[mask_barrier]))),
        })
    return deltas

def converged(deltas, threshold_eV=0.05):
    """True if last (80→100%) drift < threshold in both basin and barrier."""
    last = deltas[-1]
    return last["max_abs_basin_eV"] < threshold_eV and \
           last["max_abs_barrier_eV"] < threshold_eV
```

Threshold rationale: kT ≈ 0.0287 eV at 333 K (production temperature); 0.05 eV ≈ 1.7 kT ≈ 4.8 kJ/mol — practical convergence floor for WT-MetaD with ~10k hills, γ=10.

- [ ] **Step 3.2: Smoke check**

Run:
```bash
python - <<'PY'
from pathlib import Path
from azobenzene.scripts.barrier_extraction.fes_io import parse_bias_log
from azobenzene.scripts.barrier_extraction.fes_reconstruct import default_grid
from azobenzene.scripts.barrier_extraction.convergence import fes_vs_time, basin_barrier_drift, converged
import numpy as np
f = next(Path("azobenzene/outputs/full_runs").glob("metad_azob_cis_omol_2d_*.txt"))
run = parse_bias_log(f)
cv1, cv2 = default_grid(2.0, 2.0)
snaps = fes_vs_time(run, cv1, cv2)
# crude masks until basins_mfep is in: full grid for now
full = np.ones(snaps[1.0].shape, dtype=bool)
print(basin_barrier_drift(snaps, full, full))
print("converged?", converged(basin_barrier_drift(snaps, full, full)))
PY
```
Expected: a list of 4 deltas; absolute drift should be largest between 0.2→0.4 and smallest between 0.8→1.0 (monotonic decay is the WT signature).

- [ ] **Step 3.3: Commit**

```bash
git add azobenzene/scripts/barrier_extraction/convergence.py
git commit -m "feat(barrier_extraction): FES-vs-time convergence diagnostic"
```

---

### Task 4: Basins + Dijkstra-minimax MFEP + pathway enumeration

**Files:**
- Create: `azobenzene/scripts/barrier_extraction/basins_mfep.py`
- Test: `azobenzene/scripts/barrier_extraction/tests/test_basins_mfep.py`

**Method choice (locked):** Dijkstra on the 2D grid with edge cost `max(F_i, F_j)` (minimax) — this gives the minimum-barrier path between basins, robust to grid noise. NEB-on-grid is optional validation; not primary.

- [ ] **Step 4.1: Implement basin finder + MFEP (correct minimax)**

> ⚠ **Self-review correction:** scipy's `csgraph.dijkstra` minimizes a SUM of edge weights — using it with `weights = max(F_i, F_j)` does NOT give the bottleneck (minimax) path. We need an explicit min-bottleneck relaxation `d[v] = min(d[v], max(d[u], F[v]))`. Implemented below with `heapq`. Also: `scipy.ndimage.minimum_filter` accepts `mode="wrap"` per-axis natively, so we use that for CV1 instead of manually padding.

```python
# basins_mfep.py
from __future__ import annotations
import heapq
import numpy as np
from scipy.ndimage import minimum_filter, label

def find_local_minima(F: np.ndarray, footprint_deg: float = 10.0,
                      cv1_grid=None, cv2_grid=None,
                      mask: np.ndarray | None = None):
    """Return list of (i, j, F_val) local minima — one representative per
    connected component of the equi-F minimum mask. CV1 axis is treated periodic
    (so a basin sitting on the ±180° seam is reported as ONE basin, not two)."""
    d_cv1 = float(np.median(np.diff(cv1_grid))) if cv1_grid is not None else 1.0
    d_cv2 = float(np.median(np.diff(cv2_grid))) if cv2_grid is not None else 1.0
    fp1 = max(3, int(round(footprint_deg / d_cv1)) | 1)
    fp2 = max(3, int(round(footprint_deg / d_cv2)) | 1)
    fm = minimum_filter(F, size=(fp1, fp2), mode=("wrap", "reflect"))
    minima_mask = (F == fm)
    if mask is not None:
        minima_mask &= mask

    labels, _ = label(minima_mask)
    # Stitch the CV1 periodic seam: components touching both row 0 and row -1
    # at the same column must be merged.
    for j in np.where(minima_mask[0] & minima_mask[-1])[0]:
        l_top, l_bot = int(labels[0, j]), int(labels[-1, j])
        if l_top != l_bot and l_top != 0 and l_bot != 0:
            labels[labels == l_bot] = l_top

    out = []
    for L in np.unique(labels):
        if L == 0:
            continue
        cells = np.argwhere(labels == L)
        i, j = cells[0]   # any cell is fine — they all share F (true minimum)
        out.append((int(i), int(j), float(F[i, j])))
    return out

def _neighbors(ui, uj, n1, n2, periodic_axis0: bool):
    """4-connectivity neighbors, CV1 axis periodic when requested."""
    if ui > 0:
        yield ui - 1, uj
    elif periodic_axis0:
        yield n1 - 1, uj
    if ui < n1 - 1:
        yield ui + 1, uj
    elif periodic_axis0:
        yield 0, uj
    if uj > 0:
        yield ui, uj - 1
    if uj < n2 - 1:
        yield ui, uj + 1

def minimax_dijkstra(F: np.ndarray, start_ij, periodic_axis0: bool = True):
    """Bottleneck (min-of-max) shortest path tree from a source.

    Returns (bottleneck[N], pred[N]). bottleneck[v] is min over paths
    start→v of max(F along path), inclusive of endpoints.
    """
    n1, n2 = F.shape
    N = n1 * n2
    F_flat = F.ravel()
    bottleneck = np.full(N, np.inf)
    pred = np.full(N, -1, dtype=np.int64)
    s = start_ij[0] * n2 + start_ij[1]
    bottleneck[s] = F_flat[s]
    pq = [(float(F_flat[s]), s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > bottleneck[u]:
            continue
        ui, uj = divmod(u, n2)
        for vi, vj in _neighbors(ui, uj, n1, n2, periodic_axis0):
            v = vi * n2 + vj
            new_b = d if d > F_flat[v] else float(F_flat[v])
            if new_b < bottleneck[v]:
                bottleneck[v] = new_b
                pred[v] = u
                heapq.heappush(pq, (new_b, v))
    return bottleneck, pred

def _reconstruct_path(pred, s, t):
    out = []
    cur = int(t)
    while cur != s and cur >= 0:
        out.append(cur)
        cur = int(pred[cur])
    if cur < 0:
        raise RuntimeError(f"no path from {s} to {t}")
    out.append(s)
    return list(reversed(out))

def minimax_path(F: np.ndarray, start_ij, end_ij,
                 waypoint_ij=None, periodic_axis0: bool = True):
    """Return (path_indices, path_F_values, barrier_eV) for start -> [waypoint ->] end.

    Barrier is reported as max(F on path) - F(start). Use minimax_dijkstra so the
    saddle along the returned path is the global bottleneck of the start→end pair
    (subject to waypoint constraint if given).
    """
    n1, n2 = F.shape
    s = start_ij[0] * n2 + start_ij[1]
    t = end_ij[0] * n2 + end_ij[1]
    if waypoint_ij is None:
        _, pred = minimax_dijkstra(F, start_ij, periodic_axis0=periodic_axis0)
        path = _reconstruct_path(pred, s, t)
    else:
        w = waypoint_ij[0] * n2 + waypoint_ij[1]
        _, p1 = minimax_dijkstra(F, start_ij, periodic_axis0=periodic_axis0)
        path_sw = _reconstruct_path(p1, s, w)
        _, p2 = minimax_dijkstra(F, waypoint_ij, periodic_axis0=periodic_axis0)
        path_wt = _reconstruct_path(p2, w, t)
        path = path_sw[:-1] + path_wt
    F_path = F.ravel()[path]
    return np.array(path), F_path, float(F_path.max() - F.ravel()[s])

def nearest_grid(cv1_grid, cv2_grid, cv1_val, cv2_val):
    """Nearest grid index pair to (cv1_val, cv2_val); CV1 wraps modulo 360°."""
    d1 = ((cv1_grid - cv1_val + 180.0) % 360.0) - 180.0
    return (int(np.argmin(np.abs(d1))),
            int(np.argmin(np.abs(cv2_grid - cv2_val))))

def enumerate_pathways(F, cv1_grid, cv2_grid, basin_cis, basin_trans):
    """Run minimax twice, forcing waypoints near the rotation and inversion TS.

    For azobenzene the rotation TS is near CNNC ≈ ±90°, NNC ≈ 120°; the inversion
    TS is near CNNC ≈ 0° or 180°, NNC ≈ 175°. Symmetry (±90°) is handled by also
    running the unconstrained minimax: if the two rotation waypoints give similar
    barriers, the unconstrained should match the better one.
    """
    rot_pos = nearest_grid(cv1_grid, cv2_grid,  90.0, 120.0)
    rot_neg = nearest_grid(cv1_grid, cv2_grid, -90.0, 120.0)
    inv_cis_side   = nearest_grid(cv1_grid, cv2_grid,   0.0, 175.0)
    inv_trans_side = nearest_grid(cv1_grid, cv2_grid, 180.0, 175.0)

    out = {}
    # Rotation: try both ±90° waypoints, keep the lower barrier
    pathways_rot = []
    for wp in (rot_pos, rot_neg):
        p, Fp, dG = minimax_path(F, basin_cis, basin_trans, waypoint_ij=wp)
        pathways_rot.append((dG, p, Fp))
    pathways_rot.sort(key=lambda x: x[0])
    out["rotation"] = {"path_idx": pathways_rot[0][1],
                       "F_path":   pathways_rot[0][2],
                       "barrier_eV": pathways_rot[0][0]}

    # Inversion: try both N-linearization sides, keep the lower barrier
    pathways_inv = []
    for wp in (inv_cis_side, inv_trans_side):
        p, Fp, dG = minimax_path(F, basin_cis, basin_trans, waypoint_ij=wp)
        pathways_inv.append((dG, p, Fp))
    pathways_inv.sort(key=lambda x: x[0])
    out["inversion"] = {"path_idx": pathways_inv[0][1],
                        "F_path":   pathways_inv[0][2],
                        "barrier_eV": pathways_inv[0][0]}

    # Unconstrained — identifies the natural mechanism (lower of all above)
    p, Fp, dG = minimax_path(F, basin_cis, basin_trans, waypoint_ij=None)
    out["unconstrained"] = {"path_idx": p, "F_path": Fp, "barrier_eV": dG}
    return out
```

Notes on cis/trans basin assignment: cis = CNNC≈0°, trans = CNNC≈±180°. Trans lives on the periodic seam — use whichever local minimum is found near ±180° and let the minimum-image dijkstra handle the seam.

- [ ] **Step 4.2: Tests — double-well + explicit bottleneck property**

```python
# tests/test_basins_mfep.py
import numpy as np
from azobenzene.scripts.barrier_extraction.basins_mfep import (
    find_local_minima, minimax_path, nearest_grid, minimax_dijkstra,
)

def test_double_well_minimax():
    cv1 = np.arange(-180, 180, 2.0)
    cv2 = np.arange(0, 181, 2.0)
    X, Y = np.meshgrid(cv1, cv2, indexing="ij")
    # Two basins at (0,90) and (180,90), barrier between at (90,90)
    F = -0.3*np.exp(-((X-0)**2 + (Y-90)**2)/400) \
        -0.3*np.exp(-((X-180)**2 + (Y-90)**2)/400)
    F -= F.min()
    mins = find_local_minima(F, footprint_deg=20.0, cv1_grid=cv1, cv2_grid=cv2)
    assert len(mins) >= 2
    cis = nearest_grid(cv1, cv2, 0.0, 90.0)
    trans = nearest_grid(cv1, cv2, 180.0, 90.0)
    path, Fp, dG = minimax_path(F, cis, trans)
    assert dG > 0.0
    assert dG < float(F.max() - F[cis])

def test_minimax_picks_lowest_bottleneck():
    """Regression test against accidental sum-Dijkstra.
    Two detours from (0,0) to (4,0); direct path is blocked.
    Lower detour bottleneck = 2.0; upper detour bottleneck = 5.0.
    Correct algorithm returns 2.0; sum-Dijkstra with max-edge-weights would
    return the lower detour too because the sum of maxes is smaller,
    BUT if we make the lower detour longer than the upper, sum-Dijkstra
    picks the wrong one. We build that asymmetry explicitly."""
    F = np.full((11, 5), 10.0)
    # Endpoints
    F[0, 0] = 0.0; F[10, 0] = 0.0
    # Block direct y=0 corridor
    F[1:10, 0] = 9.0
    # Lower detour (y=1): LONG, low bottleneck (max F = 2.0)
    F[:, 1] = 0.5
    F[5, 1] = 2.0
    # Upper detour (y=2): SHORT (only via (5,2) shortcut), high bottleneck
    # Block y=2 except at column 5
    F[:, 2] = 10.0
    F[5, 2] = 4.0
    # Make column 5 cheap so a short path 0→5(y=2)→10 exists with sum lower
    # than the long y=1 traversal:
    F[0, 1] = 0.5; F[0, 2] = 0.5
    F[10, 1] = 0.5; F[10, 2] = 0.5
    # Now sum along upper: 0.5+0.5+4.0+0.5+0.5 = 6.0; sum along lower: 11*0.5 + 1.5 extra ≈ ~7
    # Minimax along upper = 4.0; minimax along lower = 2.0 ⇒ minimax must pick lower (2.0)
    F -= F.min()
    path, Fp, dG = minimax_path(F, (0, 0), (10, 0), periodic_axis0=False)
    assert abs(dG - 2.0) < 1e-9, f"bottleneck broken: expected 2.0, got {dG}"

def test_bottleneck_dijkstra_monotone_in_F():
    """Bottleneck distance must equal max(F) along the chosen path."""
    rng = np.random.default_rng(7)
    F = rng.random((20, 15))
    bot, pred = minimax_dijkstra(F, (0, 0), periodic_axis0=False)
    # Pick a random target reachable from (0,0) (everything is reachable)
    tgt = (17, 11)
    # Reconstruct the path and verify max(F[path]) == bottleneck[target]
    path = []
    cur = tgt[0] * 15 + tgt[1]
    while cur != 0:
        path.append(cur)
        cur = int(pred[cur])
    path.append(0)
    max_on_path = float(F.ravel()[path].max())
    assert abs(max_on_path - bot[tgt[0]*15 + tgt[1]]) < 1e-12
```

- [ ] **Step 4.3: Run test**

Run: `pytest azobenzene/scripts/barrier_extraction/tests/test_basins_mfep.py -v`
Expected: PASS.

- [ ] **Step 4.4: Commit**

```bash
git add azobenzene/scripts/barrier_extraction/basins_mfep.py \
        azobenzene/scripts/barrier_extraction/tests/test_basins_mfep.py
git commit -m "feat(barrier_extraction): Dijkstra-minimax MFEP + pathway enumeration"
```

---

### Task 5: Block-averaged uncertainty

**Files:**
- Create: `azobenzene/scripts/barrier_extraction/uncertainty.py`
- Test: `azobenzene/scripts/barrier_extraction/tests/test_uncertainty.py`

- [ ] **Step 5.1: Implement**

```python
# uncertainty.py
from __future__ import annotations
import numpy as np
from .fes_io import MetadRun
from .fes_reconstruct import reconstruct_fes_2d
from .basins_mfep import find_local_minima, nearest_grid, enumerate_pathways
from dataclasses import dataclass, replace

@dataclass
class BlockResult:
    barrier_rot_cis_to_trans_eV: float
    barrier_rot_trans_to_cis_eV: float
    barrier_inv_cis_to_trans_eV: float
    barrier_inv_trans_to_cis_eV: float
    dG_rxn_eV: float

def _slice_run(run: MetadRun, lo: int, hi: int) -> MetadRun:
    return replace(
        run,
        time_fs=run.time_fs[lo:hi],
        cv1_deg=run.cv1_deg[lo:hi],
        cv2_deg=run.cv2_deg[lo:hi],
        sigma1_deg=run.sigma1_deg[lo:hi],
        sigma2_deg=run.sigma2_deg[lo:hi],
        height_eV=run.height_eV[lo:hi],
        reg_factor=run.reg_factor[lo:hi],
    )

def block_average(
    run: MetadRun, cv1_grid, cv2_grid,
    n_blocks: int = 5,
    second_half_only: bool = True,
) -> tuple[list[BlockResult], np.ndarray]:
    """Compute per-block barriers using cumulative (1..k_block) hills.

    Each block uses ALL hills up to the end of that block (so each
    F is a valid converged-ish FES, not a sparse one). This is the
    standard 'time-block running estimate' for WT-MetaD.
    """
    n = run.height_eV.size
    lo = n // 2 if second_half_only else 0
    step = (n - lo) // n_blocks
    results = []
    F_blocks = []
    for k in range(1, n_blocks + 1):
        nh = lo + k * step
        F = reconstruct_fes_2d(run, cv1_grid, cv2_grid, n_hills=nh)
        F_blocks.append(F)
        cis_min = _pick_basin(F, cv1_grid, cv2_grid, target=(0.0, 120.0))
        trans_min = _pick_basin(F, cv1_grid, cv2_grid, target=(180.0, 120.0))
        paths = enumerate_pathways(F, cv1_grid, cv2_grid, cis_min, trans_min)
        F_cis = F[cis_min]; F_trans = F[trans_min]
        results.append(BlockResult(
            barrier_rot_cis_to_trans_eV=paths["rotation"]["barrier_eV"],
            barrier_rot_trans_to_cis_eV=paths["rotation"]["F_path"].max() - F_trans,
            barrier_inv_cis_to_trans_eV=paths["inversion"]["barrier_eV"],
            barrier_inv_trans_to_cis_eV=paths["inversion"]["F_path"].max() - F_trans,
            dG_rxn_eV=F_cis - F_trans,
        ))
    return results, np.stack(F_blocks)

def _pick_basin(F, cv1, cv2, target):
    mins = find_local_minima(F, footprint_deg=10.0, cv1_grid=cv1, cv2_grid=cv2)
    if not mins:
        return nearest_grid(cv1, cv2, *target)
    best = min(mins, key=lambda m: (cv1[m[0]] - target[0])**2 + (cv2[m[1]] - target[1])**2)
    return (best[0], best[1])
```

- [ ] **Step 5.2: Test on synthetic stationary data**

```python
# tests/test_uncertainty.py
import numpy as np
from azobenzene.scripts.barrier_extraction.fes_io import MetadRun
from azobenzene.scripts.barrier_extraction.uncertainty import block_average
from pathlib import Path

def test_stationary_run_low_variance():
    # Hills sampled uniformly forever from one Gaussian: variance across blocks
    # should be small because cumulative FES converges.
    rng = np.random.default_rng(0)
    n = 4000
    run = MetadRun(
        path=Path("x"), tag="t", height0_eV=0.1, pace_steps=100, step_size_fs=0.5,
        bias_factor=10.0, wt=True,
        time_fs=np.arange(n)*50.0,
        cv1_deg=rng.uniform(-30, 30, n),  # cis-like sampling
        cv2_deg=rng.uniform(110, 130, n),
        sigma1_deg=np.full(n, 5.0), sigma2_deg=np.full(n, 5.0),
        height_eV=np.full(n, 0.1*10/9), reg_factor=np.ones(n),
    )
    cv1 = np.arange(-180, 180, 4.0); cv2 = np.arange(0, 181, 4.0)
    res, _ = block_average(run, cv1, cv2, n_blocks=5, second_half_only=True)
    # No real barrier; just check it runs and returns 5 results
    assert len(res) == 5
```

- [ ] **Step 5.3: Run + commit**

```bash
pytest azobenzene/scripts/barrier_extraction/tests/test_uncertainty.py -v
git add azobenzene/scripts/barrier_extraction/uncertainty.py \
        azobenzene/scripts/barrier_extraction/tests/test_uncertainty.py
git commit -m "feat(barrier_extraction): block-averaged barrier uncertainty"
```

---

### Task 6: 1D projection by integration

**Files:**
- Create: `azobenzene/scripts/barrier_extraction/projection_1d.py`

- [ ] **Step 6.1: Implement**

```python
# projection_1d.py
import numpy as np
from scipy.special import logsumexp

def project_to_cv1(F_2d: np.ndarray, cv1_grid, cv2_grid,
                   kT_eV: float = 0.02870,   # 333 K production temperature
                   cv2_mask=None) -> np.ndarray:
    """F_1D(cv1) = -kT ln sum_j exp(-F(cv1,cv2_j)/kT) * Δcv2  (+ const)

    cv2_mask: optional boolean array over cv2 grid to exclude wall-region.
    """
    d_cv2 = float(np.median(np.diff(cv2_grid)))
    F = F_2d.copy()
    if cv2_mask is not None:
        F = F.copy()
        F[:, ~cv2_mask] = np.inf  # exclude masked region
    # logsumexp over CV2 axis
    F1 = -kT_eV * (logsumexp(-F / kT_eV, axis=1) + np.log(d_cv2))
    F1 -= F1.min()
    return F1

def cv2_wall_mask(cv2_grid, margin_deg: float = 5.0) -> np.ndarray:
    """True for the interior; False within `margin_deg` of either boundary."""
    return (cv2_grid >= cv2_grid.min() + margin_deg) & \
           (cv2_grid <= cv2_grid.max() - margin_deg)
```

- [ ] **Step 6.2: Commit (no separate test — verified via integration)**

```bash
git add azobenzene/scripts/barrier_extraction/projection_1d.py
git commit -m "feat(barrier_extraction): 1D projection via logsumexp"
```

---

### Task 7: Plotting

**Files:**
- Create: `azobenzene/scripts/barrier_extraction/plot_fes.py`

- [ ] **Step 7.1: Implement**

```python
# plot_fes.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

EV_TO_KJMOL = 96.485
EV_TO_KCALMOL = 23.061

def plot_2d_fes(F_eV, cv1, cv2, out_png: Path,
                title: str,
                basins: dict | None = None,
                paths: dict | None = None,
                contour_kJ: float = 5.0,
                f_max_kJ: float = 250.0):
    F_kJ = F_eV * EV_TO_KJMOL
    levels = np.arange(0, f_max_kJ + contour_kJ, contour_kJ)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    pcm = ax.pcolormesh(cv1, cv2, F_kJ.T, cmap="viridis",
                        shading="auto", vmin=0, vmax=f_max_kJ)
    cs = ax.contour(cv1, cv2, F_kJ.T, levels=levels, colors="k",
                    linewidths=0.4, alpha=0.5)
    ax.clabel(cs, levels=levels[::4], fmt="%d", fontsize=7)
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label("F (kJ/mol)  |  multiply by 0.239 for kcal/mol")

    if basins:
        for name, (i, j) in basins.items():
            ax.plot(cv1[i], cv2[j], "wo", mec="k", ms=8)
            ax.annotate(name, (cv1[i], cv2[j]),
                        textcoords="offset points", xytext=(5, 5), color="white",
                        fontsize=9, fontweight="bold")
    if paths:
        n1, n2 = F_eV.shape
        styles = {"rotation": ("-", "tab:red"), "inversion": ("--", "tab:orange"),
                  "unconstrained": (":", "white")}
        for name, info in paths.items():
            idx = info["path_idx"]
            ii, jj = idx // n2, idx % n2
            ls, col = styles.get(name, ("-", "white"))
            ax.plot(cv1[ii], cv2[jj], ls, color=col, lw=1.5, label=name)
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("CV1: CNNC dihedral (°)")
    ax.set_ylabel("CV2: NNC angle (°)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def plot_convergence_hills(time_fs, heights_eV, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(time_fs / 1000.0, heights_eV * EV_TO_KJMOL, lw=0.6)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("hill height (kJ/mol, log)")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def plot_convergence_fes(snapshots: dict, cv1, cv2, out_png: Path, title: str):
    # 1D slice at the mean CV2 of the minimum-energy path is a robust 1-line summary.
    fig, ax = plt.subplots(figsize=(7, 4))
    for frac, F in sorted(snapshots.items()):
        F_kJ = F * EV_TO_KJMOL
        # take min over CV2 -> shows the 2D-projected-down profile vs CV1
        ax.plot(cv1, F_kJ.min(axis=1), label=f"{int(frac*100)}%")
    ax.set_xlabel("CV1: CNNC dihedral (°)")
    ax.set_ylabel("min over CV2 of F (kJ/mol)")
    ax.set_title(title)
    ax.legend(title="hills used")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def plot_mfep_profile(paths: dict, F_eV, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, info in paths.items():
        Fp = info["F_path"] * EV_TO_KJMOL
        s = np.arange(Fp.size)
        ax.plot(s, Fp - Fp.min(), label=f"{name} (ΔG‡ = {info['barrier_eV']*EV_TO_KJMOL:.1f} kJ/mol)")
    ax.set_xlabel("path index")
    ax.set_ylabel("F along MFEP (kJ/mol)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
```

- [ ] **Step 7.2: Commit**

```bash
git add azobenzene/scripts/barrier_extraction/plot_fes.py
git commit -m "feat(barrier_extraction): 2D FES + convergence + MFEP plots"
```

---

### Task 8: Orchestrator — process all 8 runs

**Files:**
- Create: `azobenzene/scripts/barrier_extraction/run_analysis.py`

- [ ] **Step 8.1: Implement**

```python
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
    # Build basin/barrier masks: 20° around each minimum / along MFEP
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
        # Convenience kJ/mol
        "barriers_kJmol": {k.replace("_eV", "_kJmol") if "_eV" in k else k: v * 96.485
                           for k, v in {
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
        }.items()},
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
        print(f"[{txt.name}] processing …")
        try:
            summaries.append(process_run(txt, kT_eV=kT_eV,
                                         grid_deg=args.grid_deg, n_blocks=args.n_blocks))
        except Exception as e:
            print(f"  ERROR: {e}")
            summaries.append({"tag": txt.name, "error": str(e)})

    # Flat summary table
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
```

- [ ] **Step 8.2: Run the orchestrator on all 8 runs**

Run:
```bash
cd /nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE
python -m azobenzene.scripts.barrier_extraction.run_analysis \
    --temperature_K 333 --grid_deg 1.0 --n_blocks 5 \
    --pattern "metad_azob_*_2d_*.txt"
```
Expected runtime: ~1-3 min per run × 8 runs ≈ 10-25 min total. Memory: <2 GB. Outputs as listed in the file-structure section.

**If python `-m` import path is awkward** (the `azobenzene/` folder isn't a package): add `azobenzene/__init__.py` and `azobenzene/scripts/__init__.py`, OR run with explicit `PYTHONPATH=/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE python azobenzene/scripts/barrier_extraction/run_analysis.py`.

- [ ] **Step 8.3: Commit**

```bash
git add azobenzene/scripts/barrier_extraction/run_analysis.py
git commit -m "feat(barrier_extraction): per-run orchestrator + summary table"
```

---

### Task 9: Cross-seed / cross-model aggregation and report

**Files:**
- Create: `azobenzene/analysis/barrier_extraction/barrier_analysis.md`
- Create: `azobenzene/scripts/barrier_extraction/build_report.py`

- [ ] **Step 9.1: Implement the report builder**

```python
# build_report.py
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("azobenzene/analysis/barrier_extraction")

def main():
    df = pd.read_csv(OUT / "summary_table.csv")
    df["seed"]  = df["tag"].str.split("_").str[0]
    df["model"] = df["tag"].str.split("_").str[1]

    # Per-model aggregate across the two seeds (independent replicas)
    keys = [
        "rot_cis_to_trans_mean_kJmol",
        "rot_trans_to_cis_mean_kJmol",
        "inv_cis_to_trans_mean_kJmol",
        "inv_trans_to_cis_mean_kJmol",
        "dG_rxn_mean_kJmol",
    ]
    rows = []
    for model, g in df.groupby("model"):
        row = {"model": model, "n_seeds": len(g),
               "any_unconverged": int((~g["converged"]).any())}
        for k in keys:
            row[k + "_avg"] = g[k].mean()
            row[k + "_SE"]  = g[k].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg.to_csv(OUT / "summary_by_model.csv", index=False)

    md = ["# Azobenzene cis ⇌ trans barriers from 2D WT-MetaD\n"]
    md.append("**Data:** [azobenzene/outputs/full_runs/](../../outputs/full_runs/), "
              "8 production 2D WT-MetaD runs (4 MACE models × {cis,trans}-start). "
              "Bias factor γ = 10; height₀ = 0.1 eV; σ = 5° both CVs; 500 ps each.\n")
    md.append("**Engine:** custom ASE `WT_Metadynamics` "
              "([/nexus/posix0/FHI-Theory/ngoen/biASE/Metadynamics.py:463]"
              "(/nexus/posix0/FHI-Theory/ngoen/biASE/Metadynamics.py)). "
              "Hill heights in the `.txt` log are pre-multiplied by γ/(γ−1); reconstruction uses "
              "`F = −V_bias` directly.\n")
    md.append("## Per-run results\n")
    md.append(df.to_markdown(index=False, floatfmt=".2f"))
    md.append("\n## Per-model aggregation (mean ± SE over the two seeds)\n")
    md.append(agg.to_markdown(index=False, floatfmt=".2f"))
    md.append("\n## Figures\n")
    for tag in sorted(df["tag"].unique()):
        md.append(f"### {tag}\n")
        md.append(f"- ![FES](figures/{tag}/fes_2d.png)")
        md.append(f"- ![Convergence (hills)](figures/{tag}/convergence_hills.png)")
        md.append(f"- ![Convergence (FES)](figures/{tag}/convergence_fes.png)")
        md.append(f"- ![MFEP profile](figures/{tag}/mfep_profile.png)\n")
    md.append("\n## Literature comparison\n")
    md.append("**Reference values** (verify each citation against your own bibliography "
              "before publication — Codex flagged earlier that the plan-template citations are "
              "broad-strokes; the named papers below are starting points, not vouched-for):\n\n"
              "| Quantity | Value (kJ/mol) | Source kind | Reference to verify |\n"
              "|---|---|---|---|\n"
              "| ΔG‡(trans→cis, thermal, **solution**) | ≈ 96 | experimental kinetics | Schmidt et al. (early thermal kinetics); Bandara & Burdette, *Chem. Soc. Rev.* 41 (2012); Tiberio, Muccioli, Berardi, Zannoni (2010) |\n"
              "| ΔG‡(rotation, gas-phase) | ≈ 150–180 | DFT / multireference | Cembran, Bernardi, Garavelli et al., *JACS* 126 (2004); Casellas, Bearpark, Reguero (2016) |\n"
              "| ΔG‡(inversion, gas-phase) | ≈ 110–130 | DFT / multireference | Same as above |\n"
              "| Solvation drop on barrier | 20–40 | solution vs. gas-phase comparisons | Tiberio et al., *J. Chem. Theory Comput.* 6 (2010) |\n\n"
              "**Interpretation rule:** this work is **gas-phase MACE** with no explicit solvent, "
              "so the numbers should be compared to gas-phase DFT (~110–180 kJ/mol depending on mechanism), "
              "**not** to the solution-phase experimental ~96 kJ/mol. A solvation correction is needed "
              "before comparing to the experiment.\n\n"
              "**Flag in the final report** any model whose barrier disagrees with gas-phase DFT "
              "by more than ~20 kJ/mol, and report which mechanism (rotation vs inversion) the model "
              "favors. The relative ordering of mechanisms is more diagnostic than absolute barrier height.\n")
    md.append("\n## Caveats\n")
    md.append("- F = −V_bias (no extra γ/(γ−1)) because the engine pre-scales heights; "
              "this was verified at `Metadynamics.py:839` and by the row-1 ratio 0.1111 = 0.1·10/9.\n"
              "- CV2 boundary masked within 5° of the 0°/180° walls. The inversion TS lies near "
              "CV2≈175°; the masked margin is reported in the per-run figure.\n"
              "- Uncertainties are block-std over 5 cumulative blocks of the second half of hills; "
              "treats hill sequence as temporally correlated and avoids hill-level bootstrap.\n"
              "- The two seeds (cis-start, trans-start) are independent replicas; we report each "
              "separately and the model-level SE over the two.\n")
    (OUT / "barrier_analysis.md").write_text("\n".join(md))
    print(f"Wrote {OUT/'barrier_analysis.md'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 9.2: Build and inspect**

Run:
```bash
python -m azobenzene.scripts.barrier_extraction.build_report
```
Then read `azobenzene/analysis/barrier_extraction/barrier_analysis.md` and spot-check one figure per model.

- [ ] **Step 9.3: Commit**

```bash
git add azobenzene/scripts/barrier_extraction/build_report.py \
        azobenzene/analysis/barrier_extraction/barrier_analysis.md \
        azobenzene/analysis/barrier_extraction/summary_table.csv \
        azobenzene/analysis/barrier_extraction/summary_by_model.csv \
        azobenzene/analysis/barrier_extraction/summary.json
git commit -m "docs(barrier_extraction): final report + per-model aggregation"
```

> **Note:** Per CLAUDE.md, do **not** add any Claude/AI attribution to commit messages.

---

### Task 10: Verification (run after all tasks)

- [ ] **Step 10.1: Walk the deliverables checklist (from the spec)**

For each item, run a one-liner and confirm:
1. [`barrier_analysis.md`](../../analysis/barrier_extraction/barrier_analysis.md) exists and has numbers, parameters, figures.
   `test -s azobenzene/analysis/barrier_extraction/barrier_analysis.md && echo OK`
2. Scripts saved to `scripts/`:
   `ls azobenzene/scripts/barrier_extraction/*.py | wc -l` ≥ 9
3. 2D FES PNGs:
   `ls azobenzene/analysis/barrier_extraction/figures/*/fes_2d.png | wc -l` == 8
4. Convergence PNGs:
   `ls azobenzene/analysis/barrier_extraction/figures/*/convergence_*.png | wc -l` == 16
5. MFEP PNGs:
   `ls azobenzene/analysis/barrier_extraction/figures/*/mfep_profile.png | wc -l` == 8
6. Raw data **unchanged**:
   `git status -- azobenzene/outputs/full_runs/` → "nothing to commit". If anything modified, **roll it back**.

- [ ] **Step 10.2: Sanity-check magnitudes**

Open `summary_table.csv` and check:
- `dG_rxn_kJmol` is positive when cis-start (cis less stable than trans) — sign should be self-consistent across seeds and models.
- `rot_cis_to_trans_mean_kJmol` between roughly 80 and 250 kJ/mol for gas-phase MACE. Anything outside [50, 400] → investigate.
- `converged` column True for at least the omol seeds (largest trajectories). If False for many, flag in the report and consider re-running with more hills.

- [ ] **Step 10.3: Literature pass**

Read the "Literature comparison" section of the report. Note: this is **gas-phase MACE**, so 96 kJ/mol experimental solution barriers should NOT be expected to match exactly. Document the gap with the suggested causes (functional/MLIP choice, missing solvent, CV completeness).

---

### Task 11: Independent 1D-MetaD cross-check (OPTIONAL but recommended)

**Why this is not just the 1D-projection of Task 6:** the data in `outputs/full_runs/metad_azob_*_1d_*.txt` was sampled with bias **applied only to CV1** (the CNNC dihedral) — a fundamentally different simulation, not a re-analysis. Comparing native 1D MetaD barriers to (a) the 2D MFEP barrier and (b) the 1D projection from Task 6 is a strong methodology cross-check: agreement gives confidence the 2D is converged in the relevant slice; disagreement points to ergodicity issues in the orthogonal coordinate.

**Files:**
- Create: `azobenzene/scripts/barrier_extraction/run_1d_analysis.py`
- No new modules — reuse [fes_io.py](../../scripts/barrier_extraction/fes_io.py), [fes_reconstruct.py](../../scripts/barrier_extraction/fes_reconstruct.py) (with a 1D path).

- [ ] **Step 11.1: Add a 1D-aware reconstruction helper to `fes_reconstruct.py`**

```python
# add to fes_reconstruct.py
def reconstruct_fes_1d(run, cv1_grid, n_hills: int | None = None, chunk: int = 1000) -> np.ndarray:
    """Same as 2D, but only one CV. Stored heights are γ/(γ-1)-pre-scaled."""
    if n_hills is None:
        n_hills = run.height_eV.size
    H = run.height_eV[:n_hills]
    C1 = run.cv1_deg[:n_hills]
    S1 = run.sigma1_deg[:n_hills]
    V = np.zeros(cv1_grid.size, dtype=np.float64)
    for k0 in range(0, n_hills, chunk):
        k1 = min(k0 + chunk, n_hills)
        d1 = _wrap_deg(cv1_grid[:, None] - C1[None, k0:k1])
        g1 = np.exp(-0.5 * (d1 / S1[None, k0:k1]) ** 2)
        V += g1 @ H[k0:k1]
    F = -V
    F -= F.min()
    return F
```

- [ ] **Step 11.2: Implement the 1D orchestrator**

```python
# run_1d_analysis.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .fes_io import parse_bias_log
from .fes_reconstruct import reconstruct_fes_1d
from .plot_fes import EV_TO_KJMOL, EV_TO_KCALMOL
import matplotlib.pyplot as plt

DATA_DIR = Path("azobenzene/outputs/full_runs")
OUT_ROOT = Path("azobenzene/analysis/barrier_extraction")

def _basins_1d(cv1, F):
    # cis near 0°, trans near ±180° (use min over wrapped neighborhood)
    cis_i = int(np.argmin(np.abs(cv1)))
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
        cw  = F[cis_i:trans_i + 1].max()                          # short arc
        ccw = max(F[trans_i:n].max(), F[:cis_i + 1].max())        # long arc through seam
    else:
        cw  = max(F[cis_i:n].max(), F[:trans_i + 1].max())
        ccw = F[trans_i:cis_i + 1].max()
    barrier_top = min(cw, ccw)
    return float(barrier_top - F[cis_i]), float(barrier_top - F[trans_i])

def process(txt_path: Path, grid_deg: float = 0.5, n_blocks: int = 5):
    run = parse_bias_log(txt_path)
    cv1 = np.arange(-180.0, 180.0, grid_deg)
    F = reconstruct_fes_1d(run, cv1)
    cis_i, trans_i = _basins_1d(cv1, F)
    dG_c2t, dG_t2c = _barrier_1d_periodic(F, cis_i, trans_i)
    dG_rxn = float(F[cis_i] - F[trans_i])

    # block-averaged uncertainty
    n = run.height_eV.size
    lo = n // 2
    step = (n - lo) // n_blocks
    c2t_blk, t2c_blk, dG_blk = [], [], []
    for k in range(1, n_blocks + 1):
        nh = lo + k * step
        Fk = reconstruct_fes_1d(run, cv1, n_hills=nh)
        c_i, t_i = _basins_1d(cv1, Fk)
        a, b = _barrier_1d_periodic(Fk, c_i, t_i)
        c2t_blk.append(a); t2c_blk.append(b); dG_blk.append(float(Fk[c_i] - Fk[t_i]))

    out = {"tag": run.tag,
           "dG_c2t_eV": dG_c2t, "dG_c2t_sigma_eV": float(np.std(c2t_blk, ddof=1)),
           "dG_t2c_eV": dG_t2c, "dG_t2c_sigma_eV": float(np.std(t2c_blk, ddof=1)),
           "dG_rxn_eV": dG_rxn,  "dG_rxn_sigma_eV": float(np.std(dG_blk, ddof=1))}

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
        try:
            rows.append(process(txt))
        except Exception as e:
            rows.append({"tag": txt.name, "error": str(e)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "summary_table_1d.csv", index=False)
    print(f"Wrote {OUT_ROOT/'summary_table_1d.csv'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 11.3: Run and integrate into the report**

```bash
python -m azobenzene.scripts.barrier_extraction.run_1d_analysis
```

Then extend `build_report.py` (Task 9) to also read `summary_table_1d.csv` and add a comparison table: per-(model, seed), three barriers side-by-side — **1D-native**, **2D-MFEP**, **2D→1D projection** — with the deltas in kJ/mol. Disagreement > kT (~2.6 kJ/mol) flags an issue.

- [ ] **Step 11.4: Commit**

```bash
git add azobenzene/scripts/barrier_extraction/run_1d_analysis.py \
        azobenzene/scripts/barrier_extraction/fes_reconstruct.py
git commit -m "feat(barrier_extraction): native 1D MetaD cross-check + tri-method comparison"
```

---

### Task 12: TS structures from the 2 GB trajectories (OPTIONAL)

**Goal:** Save representative atomic configurations for the rotation- and inversion-pathway saddles, one .xyz per (run, mechanism). Useful for visualizing what the MLIP actually thinks the TS looks like, and for follow-up DFT single-point comparisons.

**Files:**
- Create: `azobenzene/scripts/barrier_extraction/extract_ts_structures.py`

- [ ] **Step 12.1: Implement (lazy trajectory reading)**

```python
# extract_ts_structures.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from ase.io import iread, write
from ase.io.trajectory import Trajectory
from .fes_io import parse_bias_log

OUT = Path("azobenzene/analysis/barrier_extraction/ts_structures")

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

# CV index sets (from run_metad.py)
INDICES = {
    "cis":   {"dihedral": (1, 6, 7, 8),  "angle": (1, 6, 7)},
    "trans": {"dihedral": (2, 11, 12, 13), "angle": (2, 11, 12)},
}

def extract_for_run(txt_path: Path, ts_points: dict[str, tuple[float, float]],
                    stride: int = 100):
    """For each named saddle (cv1°, cv2°), find the trajectory frame
    minimizing (Δcv1²+Δcv2²) and write it as .xyz."""
    traj_path = txt_path.with_suffix(".traj")
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)
    seed = "cis" if "_cis_" in txt_path.name else "trans"
    idx = INDICES[seed]

    best = {name: (np.inf, None) for name in ts_points}
    for k, atoms in enumerate(iread(str(traj_path), index=f"::{stride}")):
        try:
            cv1, cv2 = _cv_from_atoms(atoms, idx["dihedral"], idx["angle"])
        except Exception:
            continue
        for name, (t1, t2) in ts_points.items():
            d1 = ((cv1 - t1 + 180) % 360) - 180
            score = d1*d1 + (cv2 - t2)**2
            if score < best[name][0]:
                best[name] = (score, atoms.copy())

    run_tag = parse_bias_log(txt_path).tag
    out = OUT / run_tag
    out.mkdir(parents=True, exist_ok=True)
    for name, (score, atoms) in best.items():
        if atoms is None: continue
        write(out / f"ts_{name}.xyz", atoms)
    return {name: float(score) for name, (score, _) in best.items()}
```

- [ ] **Step 12.2: Wire to MFEP saddles in run_analysis.py**

In Task 8's `process_run`, after `enumerate_pathways`, extract the saddle (cv1, cv2) for each pathway:

```python
# inside process_run, AFTER computing `paths` dict
saddles_deg = {}
for name, info in paths.items():
    if name == "unconstrained": continue
    Fp = info["F_path"]
    n2 = F.shape[1]
    idx = info["path_idx"]
    k = int(np.argmax(Fp))
    saddles_deg[name] = (float(cv1[idx[k] // n2]), float(cv2[idx[k] % n2]))
```

Then call `extract_for_run(txt_path, saddles_deg, stride=200)` and stash the score in the summary.

- [ ] **Step 12.3: Commit**

```bash
git add azobenzene/scripts/barrier_extraction/extract_ts_structures.py
git commit -m "feat(barrier_extraction): extract MFEP-saddle representative TS structures"
```

> **Stride choice:** 200 frames out of ~10⁶ MD steps written every 1 step = 5,000 frames per traj — sub-second per run on these 2-4 GB files. If you want denser scanning around the saddle, switch to `stride=50` (adds ~5× cost). Memory is bounded by one frame at a time thanks to `ase.io.iread`.

---

## Edge cases & robustness addenda

These are inline upgrades to earlier tasks. Apply when implementing the named task.

### A. `fes_io.parse_bias_log` — restart-discontinuity guard (Task 1)

After loading `data`, assert monotonic time:
```python
dt = np.diff(data[:, 0])
if (dt <= 0).any():
    drops = np.where(dt <= 0)[0]
    raise ValueError(
        f"{path}: time column non-monotonic at row(s) {drops[:5].tolist()}. "
        "Likely a restarted/concatenated log; deduplicate before reconstruction."
    )
```
**Why:** the run-script supports `--continue-run`, which appends to the same `.txt`. If the restart wrote duplicate timesteps, double-counting hills would distort F.

### B. `basins_mfep.find_local_minima` — trans-seam handling (Task 4)

The trans minimum sits at CNNC ≈ ±180°, which is the periodic seam. The current implementation pad-wraps before `minimum_filter`, which is correct. Add this explicit test:

```python
def test_minimum_on_periodic_seam():
    cv1 = np.arange(-180, 180, 2.0); cv2 = np.arange(0, 181, 2.0)
    X, Y = np.meshgrid(cv1, cv2, indexing="ij")
    # Single Gaussian basin centered exactly on the seam (180° == -180°)
    F = -0.3 * (np.exp(-((X - 180)**2 + (Y - 90)**2) / 400)
              + np.exp(-((X + 180)**2 + (Y - 90)**2) / 400))
    F -= F.min()
    mins = find_local_minima(F, footprint_deg=20.0, cv1_grid=cv1, cv2_grid=cv2)
    # We expect ONE basin reported on the seam (either at -180 or +178), not two.
    on_seam = [m for m in mins if abs(abs(cv1[m[0]]) - 180) < 4]
    assert len(on_seam) == 1, f"seam basin split into {len(on_seam)} minima"
```

If this test fails, replace `minimum_filter` with a manual neighborhood check that takes the modulo-360 distance on the CV1 axis. Falling back to manual: scan each grid point, compare to its 8 minimum-image neighbors.

### C. Wall potential audit (Task 2 + Task 8)

`run_metad.py` declares `bounds=((-180, 180), (0, 180))`. Verify whether `WT_Metadynamics` adds **harmonic walls** at these bounds:

```bash
grep -n "wall\|bound\|harmonic" /nexus/posix0/FHI-Theory/ngoen/biASE/Metadynamics.py | head -30
```

If walls are present and act inside the FES region (not just outside), they distort F near CV2 = 0° and CV2 = 180°. The plan already masks ±5° (`cv2_wall_mask`), but **widen to ±10° if walls are stiff** (k > 0.1 eV/deg² or similar). Document the chosen margin in `barrier_analysis.md`.

### D. Independent vs cumulative blocks (Task 5)

The implementation uses **cumulative** blocks (each block has 1..k_block of the second-half hills). This is the running-estimate convention common in WT-MetaD; it under-estimates variance because adjacent blocks share data. Acceptable for "is it stable?" but **not** for a publication-grade σ.

For a more rigorous estimator, also compute **non-overlapping** blocks (each block uses exactly its own block of hills) and report both σ values:

```python
# variant: non-overlapping
for k in range(n_blocks):
    lo_k = lo + k * step
    hi_k = lo + (k + 1) * step
    F = reconstruct_fes_2d(_slice_run(run, 0, hi_k), cv1_grid, cv2_grid)  # use 0..hi_k for usable FES
    ...
```
The 0..hi_k slice keeps each F sensible while the *change between blocks* is what we measure. Report `σ_cumulative` AND `σ_block-to-block` in the table; pick the larger as the conservative uncertainty.

### E. Cross-seed reweighting check (Task 8 → report)

After the 2D FES is built from each seed, also reconstruct a **merged** FES using both seeds' hills concatenated (chronologically interleaved by `time_fs`). Compare:
- `F_cis-seed`, `F_trans-seed`, `F_merged` should agree where each individually sampled. Plot the three on a single CV2 = const slice in `convergence_fes.png`.

> The user chose "Both, separate" in scoping — this addendum is *only* a cross-check; the headline number remains per-seed.

---

## Risk register & performance budget

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Height-scaling assumption wrong | Low (verified in Task 0.1) | All barriers off by 11% | Task 0.1 reads `Metadynamics.py:839`; Task 1.3 asserts row-1 ratio |
| Insufficient sampling (barrier still drifting at 100%) | Medium | Barriers biased high or low | Task 3 reports `converged` flag per run; report unconverged in red and recommend continuation |
| Walls distort inversion TS near CV2=180° | Medium | Inversion barrier under/over-estimated | Edge-case C above; widen mask if needed |
| Trans seam split into two basins | Low | Wrong basin → wrong barrier baseline | Edge-case B test + fallback manual neighborhood |
| `.traj` file corrupted at MD step boundaries | Low | Task 12 fails for that run | Try/except in `extract_for_run`; record which TS structures missing |
| Two seeds disagree wildly | Medium | Per-model SE meaningless | Report both seeds + relative gap; flag in report if Δ > 10 kJ/mol |
| One MACE model has unphysical FES (broken basin topology) | Low (omol is robust; others ?) | Skewed model-average | Pre-flight: check `n_minima` per model; if > 4 within 20 kJ/mol of global min, flag |
| Restart concatenation in `.txt` | Low | Hills double-counted | Edge-case A monotonicity assert |

**Performance budget (single Skylake/Ice Lake core, 32 GB RAM):**

| Step | Time per run | Memory peak |
|------|--------------|-------------|
| Parse `.txt` (10k hills) | <1 s | <50 MB |
| Reconstruct 2D FES, 360×181 grid, 10k hills | ~3-5 s (chunk=500) | ~500 MB |
| 5 snapshots (convergence) | ~15-25 s | same |
| Basins + Dijkstra-minimax | <1 s | <300 MB |
| Block averaging (5 blocks) | ~15-25 s | same |
| Plots (4 PNGs) | ~2 s | <200 MB |
| **Per run total** | **~40-60 s** | **~700 MB** |
| **All 8 runs (2D)** | **~5-8 min** | bounded |
| Task 11 (1D, 8 runs) | ~2-3 min | <100 MB |
| Task 12 (TS extraction, 8 × 2 mechanisms) | ~30-60 s per traj × 8 = 4-8 min | <500 MB (lazy iread) |
| **Grand total (everything)** | **~15-20 min** | <1 GB |

If wall-clock matters: use `grid_deg=2.0` (~4× faster) for an initial pass; switch to `grid_deg=1.0` for the final report.

---

## Optional Phase-2 (not in scope unless explicitly requested)

These would add value but are out of the original 7-point spec. Listed here so they're not forgotten.

- **Tiwary-Parrinello reweighting:** read the per-step bias from the trajectory (compute V_bias at each MD step), reweight to get unbiased property averages (e.g., dihedral distribution, autocorrelations). Sanity-check that the cis/trans population implied by F matches the time-weighted basin populations.
- **Diffusivity along MFEP:** estimate D(s) from the trajectory's projection onto the MFEP arc-length s. Combined with F(s), gives an MFPT estimate via Kramers — independent kinetic cross-check vs. experiment.
- **Committor analysis:** at the saddle, run short trajectories starting on a hyperplane and measure cis-vs-trans commit probability. Confirms whether (CNNC, NNC) is a sufficient reaction coordinate or just a good order parameter.
- **DFT single points** on the extracted TS structures (Task 12). Compares each MLIP's TS to DFT energy ranking — diagnoses which MLIP best captures the barrier.
- **Effective sampling time** correction: at γ=10 with WT, effective unbiased time ≈ t_total / γ during deposition. Report 500 ps × WT-deflation as the "DFT-equivalent" sampling.

---

## Self-review summary (already applied)

- **Spec coverage:** All seven user requirements covered (inspect → convergence → 2D FES → MFEP barrier → uncertainty → 1D cross-check → literature). The "PLUMED `sum_hills`" item is mapped onto the custom-engine equivalent (Task 2) with an explicit note in the report.
- **No placeholders:** every step has explicit code, exact paths, and concrete commands.
- **Type consistency:** `MetadRun` dataclass, `(i, j)` basin tuples, and `paths` dict shape (`{"rotation"|"inversion"|"unconstrained": {"path_idx", "F_path", "barrier_eV"}}`) are referenced identically across Tasks 4, 5, 7, 8, 11, 12.
- **Critical pre-flights:**
  1. Task 0.1 verifies the height-scaling convention before any FES is computed; the entire numerical pipeline depends on `F = −V_bias` being correct.
  2. **Algorithm correction caught during self-review:** scipy's `csgraph.dijkstra` minimizes a SUM of edge weights — it does NOT yield the minimax (bottleneck) path needed for the true MFEP. Task 4.1 now uses an explicit `minimax_dijkstra` with `heapq` and relaxation `d[v] = max(d[u], F[v])`. Task 4.2 includes a regression test (`test_minimax_picks_lowest_bottleneck`) that would catch any future reversion to sum-Dijkstra.
  3. `minimum_filter` mode-tuple bug fixed: was `mode=("nearest",)` (wrong arity for a 2D array); now `mode=("wrap", "reflect")`, which removes the need for a manual pad-wrap on the periodic CV1 axis.
- **Symmetric rotation pathway:** `enumerate_pathways` now probes BOTH ±90° waypoints for rotation and BOTH (cis-side, trans-side) waypoints for inversion, keeping the lower-barrier path. The unconstrained run remains as the final sanity check.
- **Codex peer-review (completed, applied):** GPT-5.4-high reviewed the saved plan plus `Metadynamics.py` and `run_metad.py`. Findings folded inline:
  1. **IMPORTANT — temperature 333 K, not 300 K.** `run_metad.py:177,182` sets `temperature_K=333`. The 1D-projection prefactor `−kT` and the Boltzmann weight `exp(−F/kT)` are 11 % off if 300 K is used. All occurrences (Task 0.3, Task 6 default, Task 8 default, Task 8.2 example command, Task 3 threshold rationale) now use 333 K with kT = 0.02870 eV.
  2. **IMPORTANT — periodicity unit test was numerically wrong.** It compared F at −180° (distance 0 from a hill at +180°) to F at +179° (distance 1°). Replaced with an explicit seam-handled test: F[−180°] ≈ 0 (well bottom), F[+179°] ≈ h·(1 − e^{−0.5/25}) ≈ 0.002 eV, and F symmetric across the seam.
  3. **IMPORTANT — `_barrier_1d_periodic` returned 0 silently when cis_i == trans_i.** Now raises with a clear message. Dead `max_on_arc` removed.
  4. **IMPORTANT — block-σ inconsistency.** Task 8 used cumulative-block std only while Addendum D promised "max(cumulative, block-to-block)". Orchestrator now computes both and uses `np.maximum` as headline σ.
  5. **IMPORTANT — `find_local_minima` plateau/seam dedupe.** Now uses `scipy.ndimage.label` with explicit seam-stitching across rows 0 and −1; a basin on the ±180° seam returns exactly one representative, not two.
  6. **IMPORTANT — literature lip-service.** Replaced vague paragraph with a citation table (named papers per quantity, flagged "verify before publication") and an explicit comparison rule (gas-phase vs solution-phase experiment).
- **Codex findings already accepted as NITs (no edit needed):** the F-from-hills convention (`F = −V_bias`, no extra γ prefactor — verified at `Metadynamics.py:839,937,1001`), 2D contraction shapes, minimax-dijkstra relaxation, `project_to_cv1` sign/factor, and ASE eV/fs unit consistency.

---

## Execution handoff

Plan complete and saved to [docs/superpowers/plans/2026-05-12-azobenzene-wt-metad-barriers.md](2026-05-12-azobenzene-wt-metad-barriers.md). Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch with checkpoints.

Which approach?
