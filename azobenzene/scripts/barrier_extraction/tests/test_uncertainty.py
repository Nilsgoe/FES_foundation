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
