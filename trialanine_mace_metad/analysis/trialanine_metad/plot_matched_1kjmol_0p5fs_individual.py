#!/usr/bin/env python
"""Plot individual trialanine FESs from the matched 1 kJ/mol MetaD biases."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "matched_1kjmol_15deg_0p5fs_individual"
GRID_POINTS = 181  # 2 degree spacing over the periodic [-180, 180] domain.
CHUNK_SIZE = 384
PLOT_LEVELS = 18


@dataclass(frozen=True)
class Record:
    phase: str
    model: str
    slug: str
    path: Path


def records() -> list[Record]:
    amber = PROJECT / "amber_reference_metad_1kjmol_0p5fs" / "outputs"
    off24 = PROJECT / "mace_off24_m_metad_1kjmol_0p5fs" / "outputs"
    mh1 = PROJECT / "mace_mh1_metad_1kjmol_0p5fs" / "outputs"
    polar = PROJECT / "mace_polar_m_metad_1kjmol_0p5fs" / "outputs"
    return [
        Record("gas", "AMBER ff14SB", "amber", amber / "gas_amber_biase_1kjmol_0p5fs_1ns.bias"),
        Record(
            "solution",
            "AMBER ff14SB/TIP3P",
            "amber",
            amber / "solution_amber_biase_1kjmol_0p5fs_1ns.bias",
        ),
        Record(
            "gas",
            "MACE-OFF24-M",
            "mace_off24_m",
            off24 / "trialanine_gas_off24_m_1kjmol_0p5fs_1ns.metad.txt",
        ),
        Record(
            "solution",
            "MACE-OFF24-M",
            "mace_off24_m",
            off24 / "trialanine_solution_off24_m_1kjmol_0p5fs_1ns.metad.txt",
        ),
        Record(
            "gas",
            "MACE-MH1",
            "mace_mh1",
            mh1 / "trialanine_gas_mh1_1kjmol_0p5fs_1ns.metad.txt",
        ),
        Record(
            "solution",
            "MACE-MH1",
            "mace_mh1",
            mh1 / "trialanine_solution_mh1_1kjmol_0p5fs_1ns.metad.txt",
        ),
        Record(
            "gas",
            "MACE-Polar-M",
            "mace_polar_m",
            polar / "trialanine_gas_polar_m_1kjmol_0p5fs_1ns.metad.txt",
        ),
        Record(
            "solution",
            "MACE-Polar-M",
            "mace_polar_m",
            polar / "trialanine_solution_polar_m_1kjmol_0p5fs_1ns.metad.txt",
        ),
    ]


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return ((values + 180.0) % 360.0) - 180.0


def read_hills(path: Path) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            fields = line.strip().split()
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


def plot_individual(record: Record, grid: np.ndarray, fes: np.ndarray, time_fs: float) -> None:
    fig = plt.figure(figsize=(5.8, 5.1), facecolor="white")
    ax = fig.add_axes((0.14, 0.14, 0.68, 0.72))
    vmax = max(0.05, float(np.percentile(fes[np.isfinite(fes)], 99.9)))
    levels = np.linspace(0.0, vmax, PLOT_LEVELS)
    contour = ax.contourf(
        grid, grid, np.clip(fes, 0.0, vmax), levels=levels, cmap="RdBu_r", extend="max"
    )
    ax.set_title(f"{record.model} ({record.phase})\n{time_fs / 1e6:.4f} ns", pad=6)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks((-180, -90, 0, 90, 180))
    ax.set_yticks((-180, -90, 0, 90, 180))
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.tick_params(direction="in", top=True, right=True, length=4.5, width=1.0)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    colorbar_ax = fig.add_axes((0.86, 0.20, 0.035, 0.60))
    colorbar = fig.colorbar(contour, cax=colorbar_ax)
    colorbar.set_label("Relative free energy (eV)")
    colorbar.outline.set_linewidth(1.0)

    stem = f"trialanine_{record.phase}_{record.slug}_1kjmol_15deg_0p5fs"
    for extension in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{extension}", dpi=600, bbox_inches="tight", pad_inches=0.04)
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
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, str | int | float]] = []
    for record in records():
        if not record.path.exists():
            print(f"SKIP missing {record.path}")
            continue
        hills = read_hills(record.path)
        grid, fes = reconstruct(hills)
        time_fs = float(hills["time_fs"][-1])
        plot_individual(record, grid, fes, time_fs)
        summary.append(
            {
                "phase": record.phase,
                "model": record.model,
                "source": str(record.path),
                "n_hills": len(hills["time_fs"]),
                "actual_time_fs": time_fs,
                "actual_time_ns": time_fs / 1e6,
                "hill_height": "1 kJ/mol (0.01036426966 eV)",
                "sigma_deg": 15.0,
                "timestep_fs": 0.5,
                "deposition_interval_steps": 100,
                "deposition_interval_fs": 50.0,
            }
        )
        print(f"plotted {record.phase} {record.model}: {time_fs / 1e6:.5f} ns")

    if not summary:
        raise SystemExit("No matched MetaD bias files found.")
    with (OUT_DIR / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
