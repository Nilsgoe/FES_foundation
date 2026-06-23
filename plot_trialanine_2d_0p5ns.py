from __future__ import annotations

import csv
from dataclasses import dataclass
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
SUMMARY_CSV = PROJECT / "analysis" / "trialanine_metad" / "trialanine_2d_0.5ns_summary.csv"

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


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return ((values + 180.0) % 360.0) - 180.0


def periodic_delta(grid: np.ndarray, centers: np.ndarray) -> np.ndarray:
    delta = grid[None, :] - centers[:, None]
    return wrap_degrees(delta)


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
        dphi = periodic_delta(grid, data["phi"][start:stop])
        dpsi = periodic_delta(grid, data["psi"][start:stop])
        gphi = np.exp(-0.5 * (dphi / data["sigma_phi"][start:stop, None]) ** 2)
        gpsi = np.exp(-0.5 * (dpsi / data["sigma_psi"][start:stop, None]) ** 2)
        bias += np.einsum("n,nx,ny->xy", data["height"][start:stop], gphi, gpsi, optimize=True)
    fes = -bias
    fes -= np.nanmin(fes)
    return grid, fes.T


def load_surfaces() -> list[tuple[Dataset, dict[str, np.ndarray], np.ndarray, np.ndarray]]:
    surfaces = []
    for dataset in DATASETS:
        if not dataset.path.exists():
            print(f"SKIP missing {dataset.path}")
            continue
        data = truncate_metad(read_metad(dataset.path))
        grid, fes = reconstruct_fes(data)
        surfaces.append((dataset, data, grid, fes))
    return surfaces


def panel_shape(n_items: int) -> tuple[int, int]:
    return (2, 3) if n_items > 3 else (1, max(1, n_items))


def plot_phase(phase: str, surfaces: list[tuple[Dataset, dict[str, np.ndarray], np.ndarray, np.ndarray]], vmax: float) -> None:
    items = [item for item in surfaces if item[0].phase == phase]
    if not items:
        return

    nrows, ncols = panel_shape(len(items))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13.5, 8.2 if nrows == 2 else 4.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_arr = np.atleast_1d(axes).ravel()
    levels = np.linspace(0.0, vmax, 31)
    contour_ref = None
    for ax, (dataset, _data, grid, fes) in zip(axes_arr, items, strict=False):
        contour_ref = ax.contourf(grid, grid, np.clip(fes, 0.0, vmax), levels=levels, cmap="viridis", extend="max")
        ax.set_title(dataset.model)
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

    if contour_ref is not None:
        cbar = fig.colorbar(contour_ref, ax=axes_arr[: len(items)].tolist(), shrink=0.98, pad=0.015)
        cbar.set_label("Relative free energy (arb. units)")

    for ext in ("png", "pdf"):
        fig.savefig(PLOT_DIR / f"trialanine_{phase}_0p5ns_2d_model_comparison.{ext}", dpi=300)
    plt.close(fig)


def write_summary(surfaces: list[tuple[Dataset, dict[str, np.ndarray], np.ndarray, np.ndarray]]) -> None:
    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "model", "path", "n_hills", "last_time_fs", "last_time_ps"])
        for dataset, data, _grid, _fes in surfaces:
            writer.writerow([dataset.phase, dataset.model, dataset.path, len(data["time_fs"]), data["time_fs"][-1], data["time_fs"][-1] / 1000.0])


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
    surfaces = load_surfaces()
    if not surfaces:
        raise SystemExit("No trialanine MetaD files found.")
    vmax = float(np.percentile(np.concatenate([fes.ravel() for _dataset, _data, _grid, fes in surfaces]), 97.0))
    vmax = max(vmax, 1e-6)
    for phase in ("gas", "solution"):
        plot_phase(phase, surfaces, vmax)
    write_summary(surfaces)
    print(f"Wrote trialanine 0.5 ns 2D plots to {PLOT_DIR}")
    print(f"Wrote summary to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
