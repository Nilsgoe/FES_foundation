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
    assert dG <= float(F.max() - F[cis])


def test_minimax_picks_lowest_bottleneck():
    """Regression test against accidental sum-Dijkstra.
    Two detours from (0,0) to (10,0); direct path is blocked.
    Lower detour bottleneck = 2.0; upper detour bottleneck = 5.0.
    Correct algorithm returns 2.0."""
    F = np.full((11, 5), 10.0)
    F[0, 0] = 0.0; F[10, 0] = 0.0
    F[1:10, 0] = 9.0
    F[:, 1] = 0.5
    F[5, 1] = 2.0
    F[:, 2] = 10.0
    F[5, 2] = 4.0
    F[0, 1] = 0.5; F[0, 2] = 0.5
    F[10, 1] = 0.5; F[10, 2] = 0.5
    F -= F.min()
    path, Fp, dG = minimax_path(F, (0, 0), (10, 0), periodic_axis0=False)
    assert abs(dG - 2.0) < 1e-9, f"bottleneck broken: expected 2.0, got {dG}"


def test_bottleneck_dijkstra_monotone_in_F():
    """Bottleneck distance must equal max(F) along the chosen path."""
    rng = np.random.default_rng(7)
    F = rng.random((20, 15))
    bot, pred = minimax_dijkstra(F, (0, 0), periodic_axis0=False)
    tgt = (17, 11)
    path = []
    cur = tgt[0] * 15 + tgt[1]
    while cur != 0:
        path.append(cur)
        cur = int(pred[cur])
    path.append(0)
    max_on_path = float(F.ravel()[path].max())
    assert abs(max_on_path - bot[tgt[0]*15 + tgt[1]]) < 1e-12
