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
    start->v of max(F along path), inclusive of endpoints.
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
    saddle along the returned path is the global bottleneck of the start->end pair
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
    """Nearest grid index pair to (cv1_val, cv2_val); CV1 wraps modulo 360 deg."""
    d1 = ((cv1_grid - cv1_val + 180.0) % 360.0) - 180.0
    return (int(np.argmin(np.abs(d1))),
            int(np.argmin(np.abs(cv2_grid - cv2_val))))


def enumerate_pathways(F, cv1_grid, cv2_grid, basin_cis, basin_trans):
    """Run minimax twice, forcing waypoints near the rotation and inversion TS.

    For azobenzene the rotation TS is near CNNC ~ +/-90 deg, NNC ~ 120 deg; the
    inversion TS is near CNNC ~ 0 deg or 180 deg, NNC ~ 175 deg. Symmetry (+/-90 deg)
    is handled by also running the unconstrained minimax: if the two rotation waypoints
    give similar barriers, the unconstrained should match the better one.
    """
    rot_pos = nearest_grid(cv1_grid, cv2_grid,  90.0, 120.0)
    rot_neg = nearest_grid(cv1_grid, cv2_grid, -90.0, 120.0)
    inv_cis_side   = nearest_grid(cv1_grid, cv2_grid,   0.0, 175.0)
    inv_trans_side = nearest_grid(cv1_grid, cv2_grid, 180.0, 175.0)

    out = {}
    # Rotation: try both +/-90 deg waypoints, keep the lower barrier
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
