from __future__ import annotations

import csv
from dataclasses import dataclass
import heapq
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from azobenzene.scripts.barrier_extraction.basins_mfep import enumerate_pathways
from azobenzene.scripts.barrier_extraction.uncertainty import _pick_basin


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
OUT_DIR = ROOT / "azobenzene" / "analysis" / "model_comparison"
RECON_DIR = OUT_DIR / "1p5ns_reconstructed"
PATH_DIR = RECON_DIR / "paths"

MODELS = ("off", "omol", "mh1", "polar", "upet", "sol3r")
SYSTEMS = ("cis", "trans")
LABELS = {
    "off": "MACE-OFF",
    "omol": "MACE-OMOL",
    "mh1": "MACE-MH1",
    "polar": "MACE-Polar",
    "upet": "PET-SPICE",
    "sol3r": "SO3LR",
}
PATH_STYLE = {"color": "black", "linewidth": 2.4, "linestyle": "-", "label": "unconstrained path"}
PAPER_CNN_ANGLE_LIMITS = (60.0, 180.0)


@dataclass(frozen=True)
class Surface:
    system: str
    model: str
    grid1: np.ndarray
    grid2: np.ndarray
    fes: np.ndarray
    cis_min: tuple[int, int]
    trans_min: tuple[int, int]
    paths: dict[str, dict[str, np.ndarray | float]]


def load_reconstructed(system: str, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = RECON_DIR / f"azobenzene_{system}_{model}_2d_1.5ns_reconstructed_fes.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.genfromtxt(path, delimiter=",", names=True)
    cv1 = np.asarray(data["cv1"], dtype=float)
    cv2 = np.asarray(data["cv2"], dtype=float)
    free_energy = np.asarray(data["free_energy"], dtype=float)
    grid1 = np.unique(cv1)
    grid2 = np.unique(cv2)
    # Saved with cv2 as the outer loop, so transpose back to F[cv1, cv2].
    fes = free_energy.reshape(grid2.size, grid1.size).T
    return grid1, grid2, fes


def path_xy(path_idx: np.ndarray, grid1: np.ndarray, grid2: np.ndarray, fes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n2 = fes.shape[1]
    ii = path_idx // n2
    jj = path_idx % n2
    return grid1[ii], grid2[jj], fes[ii, jj]


def break_periodic_seams(x: np.ndarray, y: np.ndarray, *, period: float = 360.0) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN gaps where a periodic dihedral path crosses the plot seam.

    The minimax path correctly treats -180 and +180 as neighbors, but a normal
    line plot would connect those points across the whole panel. NaN breaks keep
    the periodic topology without drawing an artificial long segment.
    """
    if x.size < 2:
        return x, y
    jumps = np.abs(np.diff(x)) > (period / 2.0)
    if not np.any(jumps):
        return x, y

    x_out: list[float] = [float(x[0])]
    y_out: list[float] = [float(y[0])]
    for i, jump in enumerate(jumps, start=1):
        if jump:
            x_out.append(float("nan"))
            y_out.append(float("nan"))
        x_out.append(float(x[i]))
        y_out.append(float(y[i]))
    return np.asarray(x_out), np.asarray(y_out)


def neighbors_8(i: int, j: int, n1: int, n2: int):
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni = (i + di) % n1
            nj = j + dj
            if 0 <= nj < n2:
                yield ni, nj, (di * di + dj * dj) ** 0.5


def minimax_path_8_connected(F: np.ndarray, start_ij: tuple[int, int], end_ij: tuple[int, int]) -> np.ndarray:
    """Plotting path: 8-connected minimax with path length as a tie-breaker.

    The production barrier extraction uses the repo's established 4-connected
    minimax path. For visual overlays, 4-connectivity creates staircase-like
    90-degree artifacts. This keeps the same bottleneck objective but allows
    diagonal moves and selects the shorter route when several paths have the
    same maximum free energy.
    """
    n1, n2 = F.shape
    start = start_ij[0] * n2 + start_ij[1]
    end = end_ij[0] * n2 + end_ij[1]
    bottleneck = np.full(n1 * n2, np.inf)
    length = np.full(n1 * n2, np.inf)
    pred = np.full(n1 * n2, -1, dtype=np.int64)
    bottleneck[start] = float(F[start_ij])
    length[start] = 0.0
    queue = [(bottleneck[start], 0.0, start)]

    while queue:
        cost, dist, node = heapq.heappop(queue)
        if cost > bottleneck[node] or (cost == bottleneck[node] and dist > length[node]):
            continue
        if node == end:
            break
        i, j = divmod(node, n2)
        for ni, nj, step_len in neighbors_8(i, j, n1, n2):
            nxt = ni * n2 + nj
            nxt_cost = max(cost, float(F[ni, nj]))
            nxt_dist = dist + step_len
            if nxt_cost < bottleneck[nxt] or (nxt_cost == bottleneck[nxt] and nxt_dist < length[nxt]):
                bottleneck[nxt] = nxt_cost
                length[nxt] = nxt_dist
                pred[nxt] = node
                heapq.heappush(queue, (nxt_cost, nxt_dist, nxt))

    path = []
    cur = end
    while cur != start and cur >= 0:
        path.append(cur)
        cur = int(pred[cur])
    if cur < 0:
        return np.asarray([], dtype=int)
    path.append(start)
    return np.asarray(path[::-1], dtype=int)


def build_surfaces() -> list[Surface]:
    surfaces: list[Surface] = []
    PATH_DIR.mkdir(parents=True, exist_ok=True)

    for system in SYSTEMS:
        for model in MODELS:
            grid1, grid2, fes = load_reconstructed(system, model)
            cis_min = _pick_basin(fes, grid1, grid2, target=(0.0, 120.0))
            trans_min = _pick_basin(fes, grid1, grid2, target=(180.0, 120.0))
            paths = enumerate_pathways(fes, grid1, grid2, cis_min, trans_min)
            visual_path = minimax_path_8_connected(fes, cis_min, trans_min)
            if visual_path.size:
                paths["unconstrained_visual"] = {"path_idx": visual_path, "F_path": fes.ravel()[visual_path]}
            surface = Surface(system, model, grid1, grid2, fes, cis_min, trans_min, paths)
            surfaces.append(surface)
            write_paths(surface)
    return surfaces


def write_paths(surface: Surface) -> None:
    for name, info in surface.paths.items():
        x, y, f = path_xy(np.asarray(info["path_idx"], dtype=int), surface.grid1, surface.grid2, surface.fes)
        out = PATH_DIR / f"azobenzene_{surface.system}_{surface.model}_2d_1.5ns_{name}_path.csv"
        with out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path_step", "cv1_deg", "cv2_deg", "free_energy_eV"])
            writer.writerows((i, xi, yi, fi) for i, (xi, yi, fi) in enumerate(zip(x, y, f, strict=True)))


def panel_order(model: str) -> tuple[int, int]:
    model_index = MODELS.index(model)
    return model_index // 3, model_index % 3


def plot_system(system: str, surfaces: list[Surface], vmin: float, vmax: float) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), sharex=True, sharey=True, constrained_layout=True)
    levels = np.linspace(vmin, vmax, 31)
    contour_ref = None

    for surface in surfaces:
        if surface.system != system:
            continue
        row, col = panel_order(surface.model)
        ax = axes[row, col]
        contour_ref = ax.contourf(
            surface.grid1,
            surface.grid2,
            surface.fes.T,
            levels=levels,
            cmap="viridis",
            extend="max",
        )
        x, y, _ = path_xy(
            np.asarray(surface.paths.get("unconstrained_visual", surface.paths["unconstrained"])["path_idx"], dtype=int),
            surface.grid1,
            surface.grid2,
            surface.fes,
        )
        mask = (y >= PAPER_CNN_ANGLE_LIMITS[0]) & (y <= PAPER_CNN_ANGLE_LIMITS[1])
        x_plot, y_plot = break_periodic_seams(x[mask], y[mask])
        ax.plot(x_plot, y_plot, **{k: v for k, v in PATH_STYLE.items() if k != "label"})
        ax.scatter(
            [surface.grid1[surface.cis_min[0]], surface.grid1[surface.trans_min[0]]],
            [surface.grid2[surface.cis_min[1]], surface.grid2[surface.trans_min[1]]],
            s=42,
            marker="o",
            facecolors="white",
            edgecolors="black",
            linewidths=1.1,
            zorder=6,
        )
        ax.set_title(LABELS[surface.model])
        ax.set_xlim(float(surface.grid1.min()), float(surface.grid1.max()))
        ax.set_ylim(*PAPER_CNN_ANGLE_LIMITS)
        ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
        ax.set_yticks([60, 80, 100, 120, 140, 160, 180])
        ax.grid(alpha=0.15, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        if col == 0:
            ax.set_ylabel("CNN angle (deg)")
        if row == 1:
            ax.set_xlabel("CNNC dihedral (deg)")

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=PATH_STYLE["color"],
            lw=PATH_STYLE["linewidth"],
            linestyle=PATH_STYLE["linestyle"],
            label=PATH_STYLE["label"],
        )
    ]
    handles.append(
        plt.Line2D([0], [0], marker="o", markersize=7, markerfacecolor="white", markeredgecolor="black", linestyle="", label="minima")
    )
    axes[0, 0].legend(handles=handles, frameon=False, loc="lower left")

    if contour_ref is not None:
        cbar = fig.colorbar(contour_ref, ax=axes.ravel().tolist(), shrink=0.98, pad=0.015)
        cbar.set_label("Relative free energy (arb. units)")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"azobenzene_{system}_2d_model_comparison_with_paths.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 1.2,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.bbox": "tight",
        }
    )
    surfaces = build_surfaces()
    vmax = float(np.percentile(np.concatenate([s.fes.ravel() for s in surfaces]), 97.0))
    vmax = max(vmax, 1e-6)
    for system in SYSTEMS:
        plot_system(system, surfaces, vmin=0.0, vmax=vmax)
    print(f"Wrote path overlays to {OUT_DIR}")
    print(f"Wrote path CSVs to {PATH_DIR}")


if __name__ == "__main__":
    main()
