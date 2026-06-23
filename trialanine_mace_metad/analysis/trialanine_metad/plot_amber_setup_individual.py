from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "amber_setup_comparison_2026-06-11"
GRID_POINTS = 181  # 2 degree spacing over the periodic [-180, 180] domain.
CHUNK_SIZE = 384
PLOT_LEVELS = 13


@dataclass(frozen=True)
class Record:
    phase: str
    setup: str
    label: str
    path: Path
    nominal_height: str
    nominal_sigma_deg: float
    timestep_fs: float
    deposition_interval_steps: int


def records() -> list[Record]:
    original = PROJECT / "outputs" / "amber_reference_metad"
    rerun = PROJECT / "amber_reference_metad_1kjmol" / "outputs"
    return [
        Record(
            "gas",
            "original_0p1ev_sigma5",
            "AMBER: 0.1 eV, sigma 5 deg",
            original / "gas_amber_biase_prod_1ns.bias",
            "0.1 eV",
            5.0,
            0.5,
            100,
        ),
        Record(
            "solution",
            "original_0p1ev_sigma5",
            "AMBER: 0.1 eV, sigma 5 deg",
            original / "solution_amber_biase_prod_1ns.bias",
            "0.1 eV",
            5.0,
            0.5,
            100,
        ),
        Record(
            "gas",
            "1kjmol_sigma15",
            "AMBER: 1 kJ/mol, sigma 15 deg",
            rerun / "gas_amber_biase_1kjmol_1ns.bias",
            "1 kJ/mol (0.01036426966 eV)",
            15.0,
            1.0,
            100,
        ),
        Record(
            "solution",
            "1kjmol_sigma15",
            "AMBER: 1 kJ/mol, sigma 15 deg",
            rerun / "solution_amber_biase_1kjmol_1ns.bias",
            "1 kJ/mol (0.01036426966 eV)",
            15.0,
            1.0,
            100,
        ),
    ]


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return ((values + 180.0) % 360.0) - 180.0


def read_hills(path: Path) -> dict[str, np.ndarray]:
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
    return {
        "time_fs": data[:, 0],
        "phi": wrap_degrees(data[:, 1]),
        "psi": wrap_degrees(data[:, 2]),
        "sigma_phi": data[:, 3],
        "sigma_psi": data[:, 4],
        "height": data[:, 5],
    }


def reconstruct(hills: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-180.0, 180.0, GRID_POINTS)
    bias = np.zeros((GRID_POINTS, GRID_POINTS), dtype=np.float64)
    for start in range(0, len(hills["height"]), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(hills["height"]))
        dphi = wrap_degrees(grid[None, :] - hills["phi"][start:stop, None])
        dpsi = wrap_degrees(grid[None, :] - hills["psi"][start:stop, None])
        gphi = np.exp(-0.5 * (dphi / hills["sigma_phi"][start:stop, None]) ** 2)
        gpsi = np.exp(-0.5 * (dpsi / hills["sigma_psi"][start:stop, None]) ** 2)
        bias += np.einsum(
            "n,nx,ny->xy", hills["height"][start:stop], gphi, gpsi, optimize=True
        )
    fes = -bias.T
    fes -= np.nanmin(fes)
    return grid, fes


def plot_individual(
    record: Record, grid: np.ndarray, fes: np.ndarray, actual_time_fs: float
) -> None:
    fig = plt.figure(figsize=(5.8, 5.1), facecolor="white")
    ax = fig.add_axes((0.14, 0.14, 0.68, 0.72))
    vmax = max(0.05, float(np.percentile(fes[np.isfinite(fes)], 99.9)))
    levels = np.linspace(0.0, vmax, PLOT_LEVELS)
    contour = ax.contourf(grid, grid, fes, levels=levels, cmap="RdBu_r", extend="max")
    ax.set_title(f"{record.label}\n{actual_time_fs / 1e6:.4f} ns", pad=6)
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

    stem = f"trialanine_{record.phase}_amber_{record.setup}"
    for extension in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / "individual" / f"{stem}.{extension}",
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
    (OUT_DIR / "individual").mkdir(parents=True, exist_ok=True)
    source_records = records()
    missing = [record.path for record in source_records if not record.path.exists()]
    if missing:
        raise FileNotFoundError("Missing bias logs:\n" + "\n".join(map(str, missing)))

    summary: list[dict[str, str | int | float]] = []
    for record in source_records:
        hills = read_hills(record.path)
        grid, fes = reconstruct(hills)
        actual_time_fs = float(hills["time_fs"][-1])
        stem = f"trialanine_{record.phase}_amber_{record.setup}"
        np.savez_compressed(
            OUT_DIR / f"{stem}.npz",
            phi_deg=grid,
            psi_deg=grid,
            free_energy_eV=fes,
            actual_time_fs=actual_time_fs,
            source=str(record.path),
        )
        plot_individual(record, grid, fes, actual_time_fs)
        summary.append(
            {
                "phase": record.phase,
                "setup": record.setup,
                "label": record.label,
                "source": str(record.path),
                "n_hills": len(hills["time_fs"]),
                "actual_time_fs": actual_time_fs,
                "actual_time_ns": actual_time_fs / 1e6,
                "nominal_height": record.nominal_height,
                "nominal_sigma_deg": record.nominal_sigma_deg,
                "timestep_fs": record.timestep_fs,
                "deposition_interval_steps": record.deposition_interval_steps,
                "deposition_interval_fs": record.timestep_fs * record.deposition_interval_steps,
            }
        )

    with (OUT_DIR / "amber_setup_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
