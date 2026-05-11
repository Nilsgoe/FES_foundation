from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
AZOB_ANALYSIS = ROOT / "azobenzene" / "analysis"
VIPER_ANALYSIS = ROOT / "viper_analysis"
OUT_DIR = AZOB_ANALYSIS / "model_comparison"

MODELS = ("off", "omol", "mh1", "polar", "upet", "sol3r")
SYSTEMS = ("cis", "trans")
LABELS = {
    "off": "MACE-OFF",
    "omol": "MACE-OMOL",
    "mh1": "MACE-MH1",
    "polar": "MACE-Polar",
    "upet": "UPET",
    "sol3r": "SO3LR",
}
ROW_LABELS = {
    ("cis", 0): "cis",
    ("cis", 1): "cis",
    ("trans", 0): "trans",
    ("trans", 1): "trans",
}


@dataclass
class FESRecord:
    system: str
    model: str
    source: str
    path: Path


def load_records() -> list[FESRecord]:
    records: list[FESRecord] = []
    raccoon_jobs = {"off": "33975", "omol": "33976", "mh1": "33030", "polar": "33031"}
    for system in SYSTEMS:
        task = "0" if system == "cis" else "1"
        for model in ("off", "omol", "mh1", "polar"):
            records.append(
                FESRecord(
                    system=system,
                    model=model,
                    source="raccoon",
                    path=AZOB_ANALYSIS
                    / f"metad_azob_{system}_{model}_2d_raccoon_{model}_job{raccoon_jobs[model]}_task{task}_gpu1_reconstructed_fes.csv",
                )
            )
        for model in ("upet", "sol3r"):
            records.append(
                FESRecord(
                    system=system,
                    model=model,
                    source="viper",
                    path=VIPER_ANALYSIS / model / "azobenzene" / f"{system}_2d" / f"{model}_azob_{system}_2d_reconstructed_fes.csv",
                )
            )
    return [record for record in records if record.path.exists()]


def load_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    cv1 = np.asarray(data["cv1"], dtype=float)
    cv2 = np.asarray(data["cv2"], dtype=float)
    free_energy = np.asarray(data["free_energy"], dtype=float)

    grid1 = np.unique(cv1)
    grid2 = np.unique(cv2)
    fes = free_energy.reshape(grid2.size, grid1.size)
    return grid1, grid2, fes


def panel_order(record: FESRecord) -> tuple[int, int]:
    model_index = MODELS.index(record.model)
    group = model_index // 3
    col = model_index % 3
    row_base = 0 if record.system == "cis" else 2
    row = row_base + group
    return row, col


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.2,
            "savefig.bbox": "tight",
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    if not records:
        raise SystemExit("No 2D reconstructed FES files found.")

    all_max = []
    all_cv1 = []
    all_cv2 = []
    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for record in records:
        grid1, grid2, fes = load_grid(record.path)
        surfaces[(record.system, record.model)] = (grid1, grid2, fes)
        all_max.append(float(np.nanmax(fes)))
        all_cv1.extend([float(np.nanmin(grid1)), float(np.nanmax(grid1))])
        all_cv2.extend([float(np.nanmin(grid2)), float(np.nanmax(grid2))])

    vmin = 0.0
    vmax = max(all_max)
    xlim = (min(all_cv1), max(all_cv1))
    ylim = (min(all_cv2), max(all_cv2))

    fig, axes = plt.subplots(4, 3, figsize=(13.5, 16.0), sharex=True, sharey=True, constrained_layout=True)
    contour_ref = None

    for record in records:
        row, col = panel_order(record)
        ax = axes[row, col]
        grid1, grid2, fes = surfaces[(record.system, record.model)]
        contour_ref = ax.contourf(grid1, grid2, fes, levels=25, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(LABELS[record.model])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.15, linewidth=0.5)
        if col == 0:
            ax.set_ylabel("CNN angle (deg)")
        if row == 3:
            ax.set_xlabel("CNNC dihedral (deg)")

    for row in range(4):
        label = "cis" if row < 2 else "trans"
        axes[row, 0].text(
            -0.28,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )

    fig.suptitle("Azobenzene 2D MetaD comparison across models", fontsize=16, y=0.995)
    cbar = fig.colorbar(contour_ref, ax=axes, shrink=0.98, pad=0.015)
    cbar.set_label("Relative free energy (arb. units)")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"azobenzene_2d_model_comparison.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
