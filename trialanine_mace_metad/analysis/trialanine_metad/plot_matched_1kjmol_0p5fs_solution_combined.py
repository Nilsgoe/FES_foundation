#!/usr/bin/env python
"""Combined solvated-trialanine FES plot for matched 1 kJ/mol MetaD runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "matched_1kjmol_15deg_0p5fs_combined"
GRID_POINTS = 181
CHUNK_SIZE = 384
PLOT_LEVELS = 18
EV_TO_KCALMOL = 23.06054783061903


@dataclass(frozen=True)
class Record:
    model: str
    slug: str
    path: Path
    kind: str = "biase_hills"
    cv_path: Path | None = None


def records() -> list[Record]:
    amber = PROJECT / "amber_reference_metad_1kjmol_0p5fs" / "outputs"
    amber_1fs = PROJECT / "amber_reference_metad_1kjmol" / "outputs"
    off24 = PROJECT / "mace_off24_m_metad_1kjmol_0p5fs" / "outputs"
    mh1 = PROJECT / "mace_mh1_metad_1kjmol_0p5fs" / "outputs"
    polar = PROJECT / "mace_polar_m_metad_1kjmol_0p5fs" / "outputs"
    openmm = PROJECT / "openmm_amber_metad" / "outputs"
    return [
        Record("AMBER ff14SB/TIP3P\nbiASE, 0.5 fs", "amber", amber / "solution_amber_biase_1kjmol_0p5fs_1ns.bias"),
        Record(
            "AMBER ff14SB/TIP3P\nbiASE, 1 fs",
            "amber_1fs",
            amber_1fs / "solution_amber_biase_1kjmol_1ns.bias",
        ),
        Record(
            "AMBER ff14SB/TIP3P\nOpenMM, 0.5 fs",
            "openmm_amber",
            openmm / "solution_openmm_wtmetad_1kjmol_15deg_0p5fs_prod_pty.free_energy_kjmol.npy",
            "openmm_free_energy",
            openmm / "solution_openmm_wtmetad_1kjmol_15deg_0p5fs_prod_pty.cv.csv",
        ),
        Record("MACE-OFF24-M", "mace_off24_m", off24 / "trialanine_solution_off24_m_1kjmol_0p5fs_1ns.metad.txt"),
        Record("MACE-MH1", "mace_mh1", mh1 / "trialanine_solution_mh1_1kjmol_0p5fs_1ns.metad.txt"),
        Record("MACE-Polar-M", "mace_polar_m", polar / "trialanine_solution_polar_m_1kjmol_0p5fs_1ns.metad.txt"),
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


def read_openmm_free_energy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-180.0, 180.0, GRID_POINTS)
    fes = np.asarray(np.load(path), dtype=np.float64)
    if fes.shape != (GRID_POINTS, GRID_POINTS):
        raise ValueError(f"Unexpected OpenMM FES shape {fes.shape} in {path}")
    fes -= np.nanmin(fes)
    # Keep the OpenMM grid in the same phi=x, psi=y convention used for plotting.
    return grid, fes / 4.184


def read_openmm_time_fs(path: Path | None) -> tuple[int, float]:
    if path is None or not path.exists():
        return 0, float("nan")
    last_fields: list[str] | None = None
    with path.open(errors="ignore") as handle:
        next(handle, None)
        for line in handle:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 2:
                last_fields = fields
    if last_fields is None:
        return 0, float("nan")
    return int(float(last_fields[0])), float(last_fields[1])


def style_axis(ax: plt.Axes, row: int, col: int) -> None:
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks((-180, -90, 0, 90, 180))
    ax.set_yticks((-180, -90, 0, 90, 180))
    if row == 1:
        ax.set_xlabel(r"$\phi$ (deg)")
    else:
        ax.set_xlabel("")
    if col == 0:
        ax.set_ylabel(r"$\psi$ (deg)")
    else:
        ax.set_ylabel("")
    ax.tick_params(direction="in", top=True, right=True, length=4.5, width=1.0)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11.0,
            "axes.labelsize": 12.0,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items: list[tuple[Record, int, float, np.ndarray, np.ndarray]] = []
    for record in records():
        if not record.path.exists():
            print(f"SKIP missing {record.path}")
            continue
        if record.kind == "openmm_free_energy":
            grid, fes = read_openmm_free_energy(record.path)
            n_samples, time_fs = read_openmm_time_fs(record.cv_path)
        else:
            hills = read_hills(record.path)
            grid, fes = reconstruct(hills)
            fes *= EV_TO_KCALMOL
            n_samples = len(hills["time_fs"])
            time_fs = float(hills["time_fs"][-1])
        items.append((record, n_samples, time_fs, grid, fes))
        print(f"loaded {record.model}: {time_fs / 1e6:.5f} ns")

    if not items:
        raise SystemExit("No solution bias files found.")

    finite_values = np.concatenate([fes[np.isfinite(fes)].ravel() for _, _, _, _, fes in items])
    vmax = max(0.05, float(np.percentile(finite_values, 99.9)))
    levels = np.linspace(0.0, vmax, PLOT_LEVELS)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.8), sharex=True, sharey=True)
    last_contour = None
    summary: list[dict[str, str | int | float]] = []
    flat_axes = axes.ravel()
    for index, (ax, (record, n_samples, time_fs, grid, fes)) in enumerate(zip(flat_axes, items, strict=False)):
        last_contour = ax.contourf(
            grid,
            grid,
            np.clip(fes, 0.0, vmax),
            levels=levels,
            cmap="RdBu_r",
        )
        row, col = divmod(index, 3)
        style_axis(ax, row, col)
        time_ns = float(time_fs / 1e6)
        ax.set_title(f"{record.model}\n{time_ns:.4f} ns")
        summary.append(
            {
                "model": record.model,
                "source": str(record.path),
                "source_kind": record.kind,
                "n_samples": n_samples,
                "actual_time_fs": time_fs,
                "actual_time_ns": time_ns,
                "shared_vmax_kcalmol": vmax,
            }
        )
    for ax in flat_axes[len(items):]:
        ax.set_visible(False)

    if last_contour is not None:
        cbar = fig.colorbar(last_contour, ax=flat_axes[:len(items)].tolist(), location="right", shrink=0.88, pad=0.025)
        cbar.set_label("Relative free energy (kcal mol$^{-1}$)")
        cbar.ax.tick_params(direction="out", length=4, width=0.9)

    fig.suptitle("Solvated trialanine, 1 kJ/mol hills, 15 deg sigma, 0.5 fs", y=0.995)
    for extension in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"trialanine_solution_1kjmol_15deg_0p5fs_full_available_combined.{extension}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)

    with (OUT_DIR / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
