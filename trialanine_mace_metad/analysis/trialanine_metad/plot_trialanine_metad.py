#!/usr/bin/env python
"""Reconstruct trialanine MetaD Gaussian bias surfaces and plot phi/psi FES."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT / "outputs"
PLOT_DIR = PROJECT / "analysis" / "trialanine_metad" / "plots"
VIPER_ANALYSIS = PROJECT.parents[0] / "viper_analysis"

CV_MIN = -180.0
CV_MAX = 180.0
GRID_POINTS = 181
CHUNK_SIZE = 384
GAS_PAPER_MAX_TIME_FS = 1_000_000.0


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
    Dataset("solution", "MACE-OFF", OUTPUTS / "solution_off_metad_small_46949.metad.txt"),
    Dataset("solution", "MACE-MH1", OUTPUTS / "solution_mh1_metad_small_46950.metad.txt"),
    Dataset("solution", "MACE-Polar", OUTPUTS / "solution_polar_metad_small_46951.metad.txt"),
]

GAS_PAPER_DATASETS = [
    Dataset("gas", "MACE-OFF", OUTPUTS / "gas_off_metad_job44141_task0_gpu0.metad.txt"),
    Dataset("gas", "MACE-OMOL", OUTPUTS / "gas_omol_metad_job44141_task0_gpu1.metad.txt"),
    Dataset("gas", "MACE-MH1", OUTPUTS / "gas_mh1_metad_job44141_task1_gpu0.metad.txt"),
    Dataset("gas", "MACE-Polar", OUTPUTS / "gas_polar_metad_job44141_task1_gpu1.metad.txt"),
    Dataset(
        "gas",
        "PET-SPICE",
        VIPER_ANALYSIS / "upet" / "trialanine" / "gas_pet_spice" / "pet_spice_trialanine_gas_phi_psi.bias",
    ),
    Dataset(
        "gas",
        "SO3LR",
        VIPER_ANALYSIS / "sol3r" / "trialanine" / "gas" / "sol3r_trialanine_gas_phi_psi.bias",
    ),
]


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


def truncate_metad(data: dict[str, np.ndarray], max_time_fs: float) -> dict[str, np.ndarray]:
    n_rows = int(np.searchsorted(data["time_fs"], max_time_fs, side="right"))
    if n_rows <= 0:
        raise ValueError(f"No MetaD rows before {max_time_fs:g} fs")
    return {key: value[:n_rows] for key, value in data.items()}


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return ((values + 180.0) % 360.0) - 180.0


def periodic_delta(grid: np.ndarray, centers: np.ndarray) -> np.ndarray:
    delta = grid[None, :] - centers[:, None]
    return ((delta + 180.0) % 360.0) - 180.0


def reconstruct_fes(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a WT-MetaD free-energy estimate from the deposited Gaussian bias.

    The MetaD file stores the already tempered deposited heights. We reconstruct
    V(phi, psi) by summing these Gaussians with periodic dihedral distances and
    use F = -V. The logged heights are already gamma/(gamma - 1)-scaled
    by the biASE WT-MetaD implementation, as seen from the first deposited
    height 0.111111 eV for a nominal 0.1 eV hill and bias factor 10.
    """

    grid = np.linspace(CV_MIN, CV_MAX, GRID_POINTS)
    bias = np.zeros((GRID_POINTS, GRID_POINTS), dtype=np.float64)
    n_gaussians = len(data["height"])
    for start in range(0, n_gaussians, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, n_gaussians)
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
    return grid, grid, fes.T


def axis_style(ax: plt.Axes) -> None:
    ax.set_xlim(CV_MIN, CV_MAX)
    ax.set_ylim(CV_MIN, CV_MAX)
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([-150, -100, -50, 0, 50, 100, 150])
    ax.set_xlabel(r"$\phi$", fontsize=18)
    ax.set_ylabel(r"$\psi$", fontsize=18)
    ax.tick_params(labelsize=16, length=4, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def plot_surface(ax: plt.Axes, grid: np.ndarray, fes: np.ndarray, vmax: float):
    levels = np.linspace(0.0, vmax, 18)
    return ax.contourf(
        grid,
        grid,
        np.clip(fes, 0.0, vmax),
        levels=levels,
        cmap="RdBu_r",
        extend="max",
    )


def plot_individual(dataset: Dataset, grid: np.ndarray, fes: np.ndarray, data: dict[str, np.ndarray]) -> None:
    vmax = robust_vmax([fes])
    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    mesh = plot_surface(ax, grid, fes, vmax)
    axis_style(ax)
    ax.set_title(f"{dataset.model} ({dataset.phase})", fontsize=20, pad=8)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.025, fraction=0.046)
    cbar.set_label("relative FES / eV", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    add_runtime_label(ax, data)
    save_figure(fig, f"{dataset.phase}_{slug(dataset.model)}_gaussian_fes")


def plot_phase_comparison(phase: str, items: list[tuple[Dataset, dict[str, np.ndarray], np.ndarray, np.ndarray]]) -> None:
    ncols = len(items)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(5.1 * ncols, 4.9),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    if ncols == 1:
        axes = np.asarray([axes])

    vmax = robust_vmax([fes for _, _, _, fes in items])
    last_mesh = None
    for ax, (dataset, data, grid, fes) in zip(axes, items, strict=True):
        last_mesh = plot_surface(ax, grid, fes, vmax)
        axis_style(ax)
        ax.set_title(dataset.model, fontsize=20, pad=8)
        add_runtime_label(ax, data)

    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=axes.ravel().tolist(), location="top", shrink=0.7, pad=0.02)
        cbar.set_label("relative FES / eV", fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    save_figure(fig, f"{phase}_all_models_gaussian_fes")


def plot_gas_paper_comparison(items: list[tuple[Dataset, dict[str, np.ndarray], np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 8.2),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    vmax = robust_vmax([fes for _, _, _, fes in items])
    levels = np.linspace(0.0, vmax, 31)
    last_mesh = None

    for ax, (dataset, data, grid, fes) in zip(axes.ravel(), items, strict=True):
        last_mesh = ax.contourf(
            grid,
            grid,
            np.clip(fes, 0.0, vmax),
            levels=levels,
            cmap="RdBu_r",
            extend="max",
        )
        ax.set_title(dataset.model, fontsize=20, pad=8)
        ax.set_xlim(CV_MIN, CV_MAX)
        ax.set_ylim(CV_MIN, CV_MAX)
        ax.set_xticks([-100, 0, 100])
        ax.set_yticks([-150, -100, -50, 0, 50, 100, 150])
        ax.tick_params(labelsize=16, length=4, width=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\psi$", fontsize=20)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\phi$", fontsize=20)

    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=axes.ravel().tolist(), location="top", shrink=0.58, pad=0.025)
        cbar.set_label("relative FES / eV", fontsize=15)
        cbar.ax.tick_params(labelsize=12)

    save_figure(fig, "gas_1ns_all_models_gaussian_fes")


def plot_gas_paper_1ns() -> None:
    reconstructed: list[tuple[Dataset, dict[str, np.ndarray], np.ndarray, np.ndarray]] = []
    summary = ["model,path,n_hills,last_time_fs,last_time_ps,requested_cutoff_fs"]
    for dataset in GAS_PAPER_DATASETS:
        if not dataset.path.exists():
            print(f"SKIP missing {dataset.path}")
            continue
        data = truncate_metad(read_metad(dataset.path), GAS_PAPER_MAX_TIME_FS)
        grid, _, fes = reconstruct_fes(data)
        reconstructed.append((dataset, data, grid, fes))
        summary.append(
            f"{dataset.model},{dataset.path},{len(data['time_fs'])},{data['time_fs'][-1]:.2f},{data['time_fs'][-1] / 1000.0:.3f},{GAS_PAPER_MAX_TIME_FS:.2f}"
        )

    if not reconstructed:
        raise SystemExit("No gas-phase trialanine MetaD logs found for paper plot.")
    if len(reconstructed) != len(GAS_PAPER_DATASETS):
        print(f"WARNING: plotted {len(reconstructed)} of {len(GAS_PAPER_DATASETS)} requested gas datasets")

    plot_gas_paper_comparison(reconstructed)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    (PLOT_DIR / "gas_1ns_all_models_gaussian_fes_summary.csv").write_text("\n".join(summary) + "\n")


def plot_all_comparison(items: list[tuple[Dataset, dict[str, np.ndarray], np.ndarray, np.ndarray]]) -> None:
    models = ["MACE-OFF", "MACE-OMOL", "MACE-MH1", "MACE-Polar"]
    phases = ["gas", "solution"]
    lookup = {(dataset.phase, dataset.model): (dataset, data, grid, fes) for dataset, data, grid, fes in items}
    vmax = robust_vmax([fes for _, _, _, fes in items])

    fig, axes = plt.subplots(
        len(phases),
        len(models),
        figsize=(18.5, 8.9),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    last_mesh = None
    for row, phase in enumerate(phases):
        for col, model in enumerate(models):
            ax = axes[row, col]
            entry = lookup.get((phase, model))
            if entry is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes, fontsize=15)
                continue
            dataset, data, grid, fes = entry
            last_mesh = plot_surface(ax, grid, fes, vmax)
            axis_style(ax)
            if row == 0:
                ax.set_title(model, fontsize=20, pad=8)
            if col == 0:
                ax.text(
                    -0.18,
                    0.5,
                    phase.capitalize(),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=20,
                )
            add_runtime_label(ax, data)

    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=axes.ravel().tolist(), location="top", shrink=0.55, pad=0.025)
        cbar.set_label("relative FES / eV", fontsize=15)
        cbar.ax.tick_params(labelsize=12)
    save_figure(fig, "trialanine_gas_solution_all_models_gaussian_fes")


def robust_vmax(fes_list: list[np.ndarray]) -> float:
    values = np.concatenate([fes[np.isfinite(fes)].ravel() for fes in fes_list])
    vmax = float(np.percentile(values, 99.9))
    return max(vmax, 1e-6)


def add_runtime_label(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    time_ps = data["time_fs"][-1] / 1000.0
    ax.text(
        0.04,
        0.94,
        f"{time_ps:.0f} ps",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=13,
    )


def slug(text: str) -> str:
    return text.lower().replace("mace-", "").replace(" ", "_").replace("-", "_")


def save_figure(fig: plt.Figure, stem: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(PLOT_DIR / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 1.2,
            "savefig.bbox": "tight",
        }
    )

    reconstructed: list[tuple[Dataset, dict[str, np.ndarray], np.ndarray, np.ndarray]] = []
    for dataset in DATASETS:
        if not dataset.path.exists():
            print(f"SKIP missing {dataset.path}")
            continue
        data = read_metad(dataset.path)
        grid, _, fes = reconstruct_fes(data)
        reconstructed.append((dataset, data, grid, fes))
        plot_individual(dataset, grid, fes, data)
        print(f"reconstructed {dataset.phase} {dataset.model}: {dataset.path.name}")

    for phase in ("gas", "solution"):
        phase_items = [item for item in reconstructed if item[0].phase == phase]
        if phase_items:
            plot_phase_comparison(phase, phase_items)
            print(f"plotted {phase} comparison: {len(phase_items)} models")

    if reconstructed:
        plot_all_comparison(reconstructed)
        print("plotted gas/solution all-model comparison")


if __name__ == "__main__":
    main()
