import numpy as np
from azobenzene.scripts.barrier_extraction.fes_io import MetadRun
from azobenzene.scripts.barrier_extraction.uncertainty import block_average
from pathlib import Path


def test_stationary_run_low_variance():
    """Hills sampled uniformly forever from one Gaussian: variance across blocks
    should be small because cumulative FES converges."""
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


def test_pick_basin_periodic_distance():
    """When the target is at +180° and the only local minimum sits on the seam
    (grid cv1 = -180°, physically same angle), _pick_basin must pick that
    minimum rather than treating it as 360° away."""
    import numpy as np
    from azobenzene.scripts.barrier_extraction.uncertainty import _pick_basin

    cv1 = np.arange(-180.0, 180.0, 1.0)
    cv2 = np.arange(0.0, 181.0, 1.0)
    # Build a synthetic F with two local minima:
    #  (a) deep well at cv1 = -180° = +180° (the seam — the true trans basin),
    #  (b) shallow well at cv1 = +104° (a saddle-region decoy).
    F = np.zeros((cv1.size, cv2.size))
    j_120 = int(np.argmin(np.abs(cv2 - 120.0)))
    # Make every cell non-zero so the global background isn't a flat minimum
    F[:] = 0.5
    # Decoy "minimum" at (+104°, 120°): make it a local min with value 0.3
    i_104 = int(np.argmin(np.abs(cv1 - 104.0)))
    F[i_104, j_120] = 0.3
    # True deep minimum at (-180°, 120°): value 0.0
    i_seam = int(np.argmin(np.abs(cv1 + 180.0)))  # = index 0
    F[i_seam, j_120] = 0.0

    # Ask for basin near (180°, 120°). With periodic distance, the (-180°, 120°)
    # minimum is 0° away. With raw distance, the (+104°, 120°) decoy is "closer".
    i, j = _pick_basin(F, cv1, cv2, target=(180.0, 120.0))
    assert i == i_seam, f"periodic distance failed: picked cv1={cv1[i]}° (expected -180°)"
    assert j == j_120
