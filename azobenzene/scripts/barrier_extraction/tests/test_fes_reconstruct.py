import numpy as np
from azobenzene.scripts.barrier_extraction.fes_io import MetadRun
from azobenzene.scripts.barrier_extraction.fes_reconstruct import (
    reconstruct_fes_2d, default_grid, wrap_deg,
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
    assert wrap_deg(np.array([200.0]))[0] == -160.0
    assert wrap_deg(np.array([-200.0]))[0] == 160.0


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


def test_chunk_boundary_consistency():
    # Reconstruct with chunk=1, chunk=3, chunk=500 — must all agree to 1e-12
    cv1, cv2 = default_grid(4.0, 4.0)  # small grid for speed
    n_hills = 7  # crosses chunk=3 boundary (chunks of 3, 3, 1)
    rng = np.random.default_rng(0)
    run = _make_run(
        rng.uniform(-180, 180, n_hills).tolist(),
        rng.uniform(0, 180, n_hills).tolist(),
    )
    F_chunk_1   = reconstruct_fes_2d(run, cv1, cv2, chunk=1)
    F_chunk_3   = reconstruct_fes_2d(run, cv1, cv2, chunk=3)
    F_chunk_500 = reconstruct_fes_2d(run, cv1, cv2, chunk=500)
    np.testing.assert_allclose(F_chunk_1, F_chunk_3,   atol=1e-12)
    np.testing.assert_allclose(F_chunk_1, F_chunk_500, atol=1e-12)
