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
OUT_DIR = PROJECT / "analysis" / "trialanine_metad" / "1ns_6model_paper"
MAX_TIME_FS = 1_000_000.0
GRID_POINTS = 91  # 4 degree spacing over [-180, 180]
CHUNK_SIZE = 384
PLOT_LEVELS = 13

PHASES = ("gas", "solution")
MODELS = ("off", "mh1", "polar", "pet_spice", "so3lr", "amber")
LABELS = {
    "off": "MACE-OFF",
    "mh1": "MACE-MH1",
    "polar": "MACE-Polar",
    "pet_spice": "PET-SPICE",
    "so3lr": "SO3LR",
    "amber": "AMBER",
}


@dataclass(frozen=True)
class Record:
    phase: str
    model: str
    path: Path


def records() -> list[Record]:
    outputs = PROJECT / "outputs"
    viper = ROOT / "viper_analysis"
    return [
        Record("gas", "off", outputs / "gas_off_metad_job44141_task0_gpu0.metad.txt"),
        Record("gas", "mh1", outputs / "gas_mh1_metad_job44141_task1_gpu0.metad.txt"),
        Record("gas", "polar", outputs / "gas_polar_metad_job44141_task1_gpu1.metad.txt"),
        Record("gas", "pet_spice", viper / "upet/trialanine/gas_pet_spice/pet_spice_trialanine_gas_phi_psi.bias"),
        Record("gas", "so3lr", viper / "sol3r/trialanine/gas/sol3r_trialanine_gas_phi_psi.bias"),
        Record("gas", "amber", outputs / "amber_reference_metad/gas_amber_biase_prod_1ns.bias"),
        Record("solution", "off", outputs / "solution_off_metad_small_46949.metad.txt"),
        Record("solution", "mh1", outputs / "solution_mh1_metad_small_46950.metad.txt"),
        Record("solution", "polar", outputs / "solution_polar_metad_small_46951.metad.txt"),
        Record(
            "solution",
            "pet_spice",
            viper / "upet/trialanine/solution_pet_spice/pet_spice_trialanine_solution_phi_psi.bias",
        ),
        Record("solution", "so3lr", viper / "sol3r/trialanine/solution/sol3r_trialanine_solution_phi_psi.bias"),
        Record("solution", "amber", outputs / "amber_reference_metad/solution_amber_biase_prod_1ns.bias"),
    ]


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return ((values + 180.0) % 360.0) - 180.0


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
                rows.append([float(fields[index]) for index in range(8)])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No MetaD hills found in {path}")
    data = np.asarray(rows, dtype=np.float64)
    keep = np.searchsorted(data[:, 0], MAX_TIME_FS, side="right")
    data = data[:keep]
    return {
        "time_fs": data[:, 0],
        "phi": wrap_degrees(data[:, 1]),
        "psi": wrap_degrees(data[:, 2]),
        "sigma_phi": data[:, 3],
        "sigma_psi": data[:, 4],
        "height": data[:, 5],
    }


def reconstruct(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-180.0, 180.0, GRID_POINTS)
    bias = np.zeros((GRID_POINTS, GRID_POINTS), dtype=np.float64)
    for start in range(0, len(data["height"]), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(data["height"]))
        dphi = wrap_degrees(grid[None, :] - data["phi"][start:stop, None])
        dpsi = wrap_degrees(grid[None, :] - data["psi"][start:stop, None])
        gphi = np.exp(-0.5 * (dphi / data["sigma_phi"][start:stop, None]) ** 2)
        gpsi = np.exp(-0.5 * (dpsi / data["sigma_psi"][start:stop, None]) ** 2)
        bias += np.einsum("n,nx,ny->xy", data["height"][start:stop], gphi, gpsi, optimize=True)
    fes = -bias
    fes -= np.nanmin(fes)
    return grid, fes.T


def make_figure(
    phase: str,
    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float]],
    vmax: float,
) -> None:
    fig = plt.figure(figsize=(12.0, 8.2), facecolor="white")
    panel_width = 0.245
    panel_height = 0.385
    x_positions = (0.055, 0.355, 0.655)
    y_positions = (0.555, 0.075)
    positions = tuple((x, y, panel_width, panel_height) for y in y_positions for x in x_positions)
    levels = np.linspace(0.0, vmax, PLOT_LEVELS)
    contour = None

    for panel_index, (model, position) in enumerate(zip(MODELS, positions, strict=True)):
        ax = fig.add_axes(position)
        grid, fes, actual_time_fs = surfaces[(phase, model)]
        contour = ax.contourf(grid, grid, fes, levels=levels, cmap="RdBu_r", extend="max")
        ax.set_title(f"{LABELS[model]}\n{actual_time_fs / 1e6:.2f} ns", pad=6)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks((-180, -90, 0, 90, 180))
        ax.set_yticks((-180, -90, 0, 90, 180))
        ax.set_xlabel(r"$\phi$ (deg)")
        ax.set_ylabel(r"$\psi$ (deg)" if panel_index in (0, 3) else "")
        ax.tick_params(direction="in", top=True, right=True, length=4.5, width=1.0)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5, width=0.8)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    colorbar_ax = fig.add_axes((0.94, 0.17, 0.018, 0.68))
    colorbar = fig.colorbar(contour, cax=colorbar_ax)
    colorbar.set_label("Relative free energy (eV)")
    colorbar.outline.set_linewidth(1.0)
    colorbar.ax.tick_params(direction="out", length=4, width=0.9)

    for extension in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"trialanine_{phase}_2d_6model_paper.{extension}",
            dpi=600,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def make_individual_figure(
    phase: str,
    model: str,
    grid: np.ndarray,
    fes: np.ndarray,
    actual_time_fs: float,
) -> None:
    fig = plt.figure(figsize=(5.8, 5.1), facecolor="white")
    ax = fig.add_axes((0.14, 0.14, 0.68, 0.72))
    vmax = float(np.ceil(np.nanmax(fes) * 20.0) / 20.0)
    levels = np.linspace(0.0, vmax, PLOT_LEVELS)
    contour = ax.contourf(grid, grid, fes, levels=levels, cmap="RdBu_r", extend="max")
    ax.set_title(f"{LABELS[model]}\n{actual_time_fs / 1e6:.2f} ns", pad=6)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks((-180, -90, 0, 90, 180))
    ax.set_yticks((-180, -90, 0, 90, 180))
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.tick_params(direction="in", top=True, right=True, length=4.5, width=1.0)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5, width=0.8)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    colorbar_ax = fig.add_axes((0.86, 0.20, 0.035, 0.60))
    colorbar = fig.colorbar(contour, cax=colorbar_ax)
    colorbar.set_label("Relative free energy (eV)")
    colorbar.outline.set_linewidth(1.0)
    colorbar.ax.tick_params(direction="out", length=4, width=0.9)

    for extension in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / "individual" / f"trialanine_{phase}_{model}_2d_paper.{extension}",
            dpi=600,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 1.1,
            "axes.titleweight": "medium",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "individual").mkdir(parents=True, exist_ok=True)
    source_records = records()
    missing = [record.path for record in source_records if not record.path.exists()]
    if missing:
        raise FileNotFoundError("Missing bias logs:\n" + "\n".join(map(str, missing)))

    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float]] = {}
    summary: list[dict[str, str | int | float]] = []
    maxima = []
    for record in source_records:
        data = read_metad(record.path)
        grid, fes = reconstruct(data)
        actual_time_fs = float(data["time_fs"][-1])
        surfaces[(record.phase, record.model)] = (grid, fes, actual_time_fs)
        maxima.append(float(np.nanmax(fes)))
        np.savez_compressed(
            OUT_DIR / f"trialanine_{record.phase}_{record.model}_2d.npz",
            phi_deg=grid,
            psi_deg=grid,
            free_energy_eV=fes,
            actual_time_fs=actual_time_fs,
            source=str(record.path),
        )
        summary.append(
            {
                "phase": record.phase,
                "model": record.model,
                "label": LABELS[record.model],
                "source": str(record.path),
                "n_hills": len(data["time_fs"]),
                "actual_time_fs": actual_time_fs,
                "actual_time_ns": actual_time_fs / 1e6,
            }
        )

    with (OUT_DIR / "trialanine_2d_6model_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    vmax = float(np.ceil(max(maxima) * 10.0) / 10.0)
    for phase in PHASES:
        make_figure(phase, surfaces, vmax)
        for model in MODELS:
            grid, fes, actual_time_fs = surfaces[(phase, model)]
            make_individual_figure(phase, model, grid, fes, actual_time_fs)


if __name__ == "__main__":
    main()
