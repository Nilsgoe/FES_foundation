from __future__ import annotations

import csv
from dataclasses import dataclass
import heapq
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
PROJECT = ROOT / "trialanine_mace_metad"
OUTPUTS = PROJECT / "outputs"
VIPER_ANALYSIS = ROOT / "viper_analysis"
PLOT_DIR = PROJECT / "analysis" / "trialanine_metad" / "plots"
PATH_DIR = PROJECT / "analysis" / "trialanine_metad" / "paths_0p5ns"

MAX_TIME_FS = 500_000.0
CV_MIN = -180.0
CV_MAX = 180.0
GRID_POINTS = 181
CHUNK_SIZE = 384


@dataclass(frozen=True)
class Dataset:
    phase: str
    model: str
    path: Path


DATASETS = [
    Dataset("gas", "MACE-OFF", OUTPUTS / "gas_off_metad_job44141_task0_gpu0.metad.txt"),
    Dataset("gas", "MACE-OMOL", OUTPUTS / "gas_omol_metad_job44141_task0_gpu1.metad.txt"),
    Dataset("gas", "MACE-MH1", OUTPUTS / "gas_mh1_metad_job44141_task1_gpu0.metad.txt"),
    Dataset("gas", "MACE-Polar", OUTPUTS / "gas_polar_metad_job44141_task1_gpu1.metad.txt"),
    Dataset("gas", "PET-SPICE", VIPER_ANALYSIS / "upet" / "trialanine" / "gas_pet_spice" / "pet_spice_trialanine_gas_phi_psi.bias"),
    Dataset("gas", "SO3LR", VIPER_ANALYSIS / "sol3r" / "trialanine" / "gas" / "sol3r_trialanine_gas_phi_psi.bias"),
    Dataset("solution", "MACE-OFF", OUTPUTS / "solution_off_metad_small_46949.metad.txt"),
    Dataset("solution", "MACE-MH1", OUTPUTS / "solution_mh1_metad_small_46950.metad.txt"),
    Dataset("solution", "MACE-Polar", OUTPUTS / "solution_polar_metad_small_46951.metad.txt"),
    Dataset("solution", "PET-SPICE", VIPER_ANALYSIS / "upet" / "trialanine" / "solution_pet_spice" / "pet_spice_trialanine_solution_phi_psi.bias"),
    Dataset("solution", "SO3LR", VIPER_ANALYSIS / "sol3r" / "trialanine" / "solution" / "sol3r_trialanine_solution_phi_psi.bias"),
]


@dataclass(frozen=True)
class Surface:
    dataset: Dataset
    grid: np.ndarray
    fes: np.ndarray
    start_min: tuple[int, int]
    end_min: tuple[int, int]
    path_idx: np.ndarray
    n_hills: int
    last_time_fs: float


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return ((values + 180.0) % 360.0) - 180.0


def periodic_delta(grid: np.ndarray, centers: np.ndarray) -> np.ndarray:
    delta = grid[None, :] - centers[:, None]
    return wrap_degrees(delta)


def periodic_distance(a: float, b: float) -> float:
    return abs(float(wrap_degrees(np.asarray([a - b]))[0]))


def read_metad(path: Path) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("!") or stripped.startswith("time"):
                continue
            fields = stripped.split()
            if len(fields) < 8:
                continue
            try:
                rows.append([float(fields[i]) for i in range(8)])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No MetaD rows could be read from {path}")
    data = np.asarray(rows, dtype=np.float64)
    return {
        "time_fs": data[:, 0],
        "phi": wrap_degrees(data[:, 1]),
        "psi": wrap_degrees(data[:, 2]),
        "sigma_phi": data[:, 3],
        "sigma_psi": data[:, 4],
        "height": data[:, 5],
        "bias_factor": data[:, 6],
    }


def truncate_metad(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n_rows = int(np.searchsorted(data["time_fs"], MAX_TIME_FS, side="right"))
    if n_rows <= 0:
        raise ValueError(f"No MetaD rows before {MAX_TIME_FS:g} fs")
    return {key: value[:n_rows] for key, value in data.items()}


def reconstruct_fes(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(CV_MIN, CV_MAX, GRID_POINTS)
    bias = np.zeros((GRID_POINTS, GRID_POINTS), dtype=np.float64)
    for start in range(0, len(data["height"]), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(data["height"]))
        phi = data["phi"][start:stop]
        psi = data["psi"][start:stop]
        sigma_phi = data["sigma_phi"][start:stop]
        sigma_psi = data["sigma_psi"][start:stop]
        height = data["height"][start:stop]
        dphi = periodic_delta(grid, phi)
        dpsi = periodic_delta(grid, psi)
        gphi = np.exp(-0.5 * (dphi / sigma_phi[:, None]) ** 2)
        gpsi = np.exp(-0.5 * (dpsi / sigma_psi[:, None]) ** 2)
        bias += np.einsum("n,nx,ny->xy", height, gphi, gpsi, optimize=True)
    fes = -bias
    fes -= np.nanmin(fes)
    return grid, fes


def local_minima(F: np.ndarray) -> list[tuple[int, int, float]]:
    minima: list[tuple[int, int, float]] = []
    n1, n2 = F.shape
    for i in range(n1):
        for j in range(n2):
            val = F[i, j]
            is_min = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    if F[(i + di) % n1, (j + dj) % n2] < val:
                        is_min = False
                        break
                if not is_min:
                    break
            if is_min:
                minima.append((i, j, float(val)))
    minima.sort(key=lambda x: x[2])
    return minima


def pick_two_lowest_separated(F: np.ndarray, grid: np.ndarray, min_sep_deg: float = 45.0) -> tuple[tuple[int, int], tuple[int, int]]:
    minima = local_minima(F)
    if not minima:
        flat = np.argsort(F.ravel())
        return divmod(int(flat[0]), F.shape[1]), divmod(int(flat[1]), F.shape[1])
    first = minima[0]
    for candidate in minima[1:]:
        dphi = periodic_distance(grid[candidate[0]], grid[first[0]])
        dpsi = periodic_distance(grid[candidate[1]], grid[first[1]])
        if (dphi * dphi + dpsi * dpsi) ** 0.5 >= min_sep_deg:
            return (first[0], first[1]), (candidate[0], candidate[1])
    flat = np.argsort(F.ravel())
    return (first[0], first[1]), divmod(int(flat[min(1, flat.size - 1)]), F.shape[1])


def neighbors_8(i: int, j: int, n1: int, n2: int):
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            yield (i + di) % n1, (j + dj) % n2, (di * di + dj * dj) ** 0.5


def minimax_path_8_connected(F: np.ndarray, start_ij: tuple[int, int], end_ij: tuple[int, int]) -> np.ndarray:
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


def path_xy(path_idx: np.ndarray, grid: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i, j = np.divmod(path_idx, F.shape[1])
    return grid[i], grid[j], F[i, j]


def break_periodic_seams(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size < 2:
        return x, y
    jumps = (np.abs(np.diff(x)) > 180.0) | (np.abs(np.diff(y)) > 180.0)
    if not np.any(jumps):
        return x, y
    x_out = [float(x[0])]
    y_out = [float(y[0])]
    for i, jump in enumerate(jumps, start=1):
        if jump:
            x_out.append(float("nan"))
            y_out.append(float("nan"))
        x_out.append(float(x[i]))
        y_out.append(float(y[i]))
    return np.asarray(x_out), np.asarray(y_out)


def build_surfaces() -> list[Surface]:
    PATH_DIR.mkdir(parents=True, exist_ok=True)
    surfaces: list[Surface] = []
    for dataset in DATASETS:
        if not dataset.path.exists():
            print(f"SKIP missing {dataset.path}")
            continue
        data = truncate_metad(read_metad(dataset.path))
        grid, F = reconstruct_fes(data)
        start_min, end_min = pick_two_lowest_separated(F, grid)
        path_idx = minimax_path_8_connected(F, start_min, end_min)
        surface = Surface(dataset, grid, F, start_min, end_min, path_idx, len(data["time_fs"]), float(data["time_fs"][-1]))
        surfaces.append(surface)
        write_path(surface)
    return surfaces


def write_path(surface: Surface) -> None:
    x, y, f = path_xy(surface.path_idx, surface.grid, surface.fes)
    out = PATH_DIR / f"trialanine_{surface.dataset.phase}_{slug(surface.dataset.model)}_0.5ns_unconstrained_path.csv"
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_step", "phi_deg", "psi_deg", "free_energy_eV"])
        writer.writerows((i, xi, yi, fi) for i, (xi, yi, fi) in enumerate(zip(x, y, f, strict=True)))


def slug(text: str) -> str:
    return text.lower().replace("mace-", "").replace(" ", "_").replace("-", "_")


def panel_shape(n_items: int) -> tuple[int, int]:
    return (2, 3) if n_items > 3 else (1, max(1, n_items))


def plot_phase(phase: str, surfaces: list[Surface], vmax: float) -> None:
    items = [surface for surface in surfaces if surface.dataset.phase == phase]
    if not items:
        return
    nrows, ncols = panel_shape(len(items))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 8.2 if nrows == 2 else 4.6), sharex=True, sharey=True, constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    levels = np.linspace(0.0, vmax, 31)
    contour_ref = None
    for ax, surface in zip(axes_arr, items, strict=False):
        contour_ref = ax.contourf(surface.grid, surface.grid, surface.fes.T, levels=levels, cmap="viridis", extend="max")
        x, y, _ = path_xy(surface.path_idx, surface.grid, surface.fes)
        x_plot, y_plot = break_periodic_seams(x, y)
        ax.plot(x_plot, y_plot, color="black", linewidth=2.4)
        ax.scatter(
            [surface.grid[surface.start_min[0]], surface.grid[surface.end_min[0]]],
            [surface.grid[surface.start_min[1]], surface.grid[surface.end_min[1]]],
            s=42,
            marker="o",
            facecolors="white",
            edgecolors="black",
            linewidths=1.1,
            zorder=6,
        )
        ax.set_title(surface.dataset.model)
        ax.set_xlim(CV_MIN, CV_MAX)
        ax.set_ylim(CV_MIN, CV_MAX)
        ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
        ax.set_yticks([-150, -100, -50, 0, 50, 100, 150])
        ax.grid(alpha=0.15, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    for ax in axes_arr[len(items):]:
        ax.axis("off")
    for idx, ax in enumerate(axes_arr[: len(items)]):
        row = idx // ncols
        col = idx % ncols
        if col == 0:
            ax.set_ylabel(r"$\psi$ (deg)")
        if row == nrows - 1:
            ax.set_xlabel(r"$\phi$ (deg)")
    handles = [
        plt.Line2D([0], [0], color="black", lw=2.4, label="unconstrained path"),
        plt.Line2D([0], [0], marker="o", markersize=7, markerfacecolor="white", markeredgecolor="black", linestyle="", label="minima"),
    ]
    axes_arr[0].legend(handles=handles, frameon=False, loc="lower left")
    if contour_ref is not None:
        cbar = fig.colorbar(contour_ref, ax=axes_arr[: len(items)].tolist(), shrink=0.98, pad=0.015)
        cbar.set_label("Relative free energy (arb. units)")
    for ext in ("png", "pdf"):
        fig.savefig(PLOT_DIR / f"trialanine_{phase}_0p5ns_model_comparison_with_paths.{ext}", dpi=300)
    plt.close(fig)


def write_summary(surfaces: list[Surface]) -> None:
    out = PATH_DIR / "trialanine_0.5ns_path_summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "model", "path", "n_hills", "last_time_fs", "last_time_ps", "start_phi", "start_psi", "end_phi", "end_psi"])
        for surface in surfaces:
            writer.writerow(
                [
                    surface.dataset.phase,
                    surface.dataset.model,
                    surface.dataset.path,
                    surface.n_hills,
                    surface.last_time_fs,
                    surface.last_time_fs / 1000.0,
                    surface.grid[surface.start_min[0]],
                    surface.grid[surface.start_min[1]],
                    surface.grid[surface.end_min[0]],
                    surface.grid[surface.end_min[1]],
                ]
            )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.2,
            "savefig.bbox": "tight",
        }
    )
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    surfaces = build_surfaces()
    if not surfaces:
        raise SystemExit("No trialanine MetaD files found.")
    vmax = float(np.percentile(np.concatenate([surface.fes.ravel() for surface in surfaces]), 97.0))
    vmax = max(vmax, 1e-6)
    for phase in ("gas", "solution"):
        plot_phase(phase, surfaces, vmax)
    write_summary(surfaces)
    print(f"Wrote trialanine path plots to {PLOT_DIR}")
    print(f"Wrote trialanine path CSVs to {PATH_DIR}")


if __name__ == "__main__":
    main()
