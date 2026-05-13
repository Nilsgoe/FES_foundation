from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace
from .fes_io import MetadRun
from .fes_reconstruct import reconstruct_fes_2d
from .basins_mfep import find_local_minima, nearest_grid, enumerate_pathways


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


def _pick_basin(F, cv1, cv2, target):
    """Find local minimum nearest to target (cv1°, cv2°). Fall back to
    nearest_grid if find_local_minima yields nothing."""
    mins = find_local_minima(F, footprint_deg=10.0, cv1_grid=cv1, cv2_grid=cv2)
    if not mins:
        return nearest_grid(cv1, cv2, *target)
    best = min(mins, key=lambda m: (cv1[m[0]] - target[0])**2 + (cv2[m[1]] - target[1])**2)
    return (best[0], best[1])


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
