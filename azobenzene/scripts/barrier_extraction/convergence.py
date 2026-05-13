"""FES convergence diagnostics for WT-MetaD runs.

Threshold rationale:
    Default threshold 0.05 eV ≈ 1.7 × kT at 333 K (the production temperature,
    kT = 0.02870 eV) ≈ 4.8 kJ/mol — practical convergence floor for WT-MetaD
    with γ=10 and ~10k hills.
"""
from __future__ import annotations

import numpy as np

from .fes_io import MetadRun
from .fes_reconstruct import reconstruct_fes_2d


def fes_vs_time(
    run: MetadRun,
    cv1: np.ndarray,
    cv2: np.ndarray,
    fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
) -> dict[float, np.ndarray]:
    """Reconstruct the FES at multiple time fractions of the run.

    Parameters
    ----------
    run:
        Parsed MetaD run from ``parse_bias_log``.
    cv1, cv2:
        Grid arrays for the two collective variables.
    fractions:
        Sequence of floats in (0, 1] at which to snapshot the FES.

    Returns
    -------
    dict mapping each fraction to an F(cv1, cv2) array (eV, min-shifted).
    """
    n = run.height_eV.size
    out: dict[float, np.ndarray] = {}
    for f in fractions:
        nh = max(1, int(round(n * f)))
        out[f] = reconstruct_fes_2d(run, cv1, cv2, n_hills=nh)
    return out


def basin_barrier_drift(
    fes_snapshots: dict[float, np.ndarray],
    mask_basin: np.ndarray,
    mask_barrier: np.ndarray,
) -> list[dict]:
    """Compute the max |ΔF| between consecutive FES snapshots over two masks.

    Parameters
    ----------
    fes_snapshots:
        Output of ``fes_vs_time``: dict {fraction: F array}.
    mask_basin:
        Boolean array (same shape as F) selecting basin grid points.
    mask_barrier:
        Boolean array (same shape as F) selecting barrier grid points.

    Returns
    -------
    List of dicts, one per consecutive fraction pair, each with keys:
      ``from``, ``to``, ``max_abs_basin_eV``, ``max_abs_barrier_eV``.
    """
    fracs = sorted(fes_snapshots)
    deltas = []
    for a, b in zip(fracs[:-1], fracs[1:]):
        d = fes_snapshots[b] - fes_snapshots[a]
        deltas.append({
            "from": a,
            "to": b,
            "max_abs_basin_eV":   float(np.max(np.abs(d[mask_basin]))),
            "max_abs_barrier_eV": float(np.max(np.abs(d[mask_barrier]))),
        })
    return deltas


def converged(deltas: list[dict], threshold_eV: float = 0.05) -> bool:
    """Return True if the last (80 → 100 %) drift is below *threshold_eV*.

    Both basin and barrier max |ΔF| must be below the threshold.

    Parameters
    ----------
    deltas:
        Output of ``basin_barrier_drift``.
    threshold_eV:
        Convergence threshold in eV (default 0.05 eV, see module docstring).
    """
    last = deltas[-1]
    return (
        last["max_abs_basin_eV"] < threshold_eV
        and last["max_abs_barrier_eV"] < threshold_eV
    )
