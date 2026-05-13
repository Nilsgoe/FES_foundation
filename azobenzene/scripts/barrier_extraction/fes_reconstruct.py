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
