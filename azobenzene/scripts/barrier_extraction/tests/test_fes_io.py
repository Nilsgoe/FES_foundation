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
    assert np.isclose(run.height_eV[0], 0.1 * 10 / 9, rtol=1e-4), (
        f"height pre-scaling broken: got {run.height_eV[0]}"
    )
    # Hills should decay (well-tempered)
    assert run.height_eV[-100:].mean() < run.height_eV[:100].mean()
