from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from azobenzene.scripts.barrier_extraction.fes_io import parse_bias_log
from azobenzene.scripts.barrier_extraction.fes_reconstruct import default_grid, reconstruct_fes_2d


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
OUT_DIR = ROOT / "azobenzene" / "analysis" / "model_comparison" / "2ns_8model_paper"
MAX_TIME_FS = 2_000_000.0
PLOT_LEVELS = 13

SYSTEMS = ("cis", "trans")
MODELS = ("off", "mh1", "polar", "pet_spice", "so3lr", "scratch", "ft_so3lr", "ft_mh1")
LABELS = {
    "off": "MACE-OFF",
    "mh1": "MACE-MH1",
    "polar": "MACE-Polar",
    "pet_spice": "PET-SPICE",
    "so3lr": "SO3LR",
    "scratch": "MACE from scratch",
    "ft_so3lr": "Fine-tuned SO3LR",
    "ft_mh1": "Fine-tuned MACE-MH1",
}


@dataclass(frozen=True)
class Record:
    system: str
    model: str
    path: Path


def find_pet_spice(system: str) -> Path:
    current = (
        Path("/work/gpuviper_ptmp/Enhanced_sampling/upet/azobenzene")
        / f"pet_spice_{system}_2d"
        / "outputs"
        / f"pet_spice_azob_{system}_2d.bias"
    )
    fallback = (
        ROOT
        / "viper_analysis"
        / "upet"
        / "azobenzene"
        / f"pet_spice_{system}_2d"
        / f"pet_spice_azob_{system}_2d.bias"
    )
    return current if current.exists() else fallback


def load_records() -> list[Record]:
    jobs = {"off": ("off", "33975"), "mh1": ("mh1", "33030"), "polar": ("polar", "33031")}
    records: list[Record] = []
    for system in SYSTEMS:
        task = "0" if system == "cis" else "1"
        for model, (filename_model, job) in jobs.items():
            records.append(
                Record(
                    system,
                    model,
                    ROOT
                    / "azobenzene"
                    / "outputs"
                    / f"metad_azob_{system}_{filename_model}_2d_raccoon_{filename_model}_job{job}_task{task}_gpu1.txt",
                )
            )
        records.extend(
            [
                Record(system, "pet_spice", find_pet_spice(system)),
                Record(
                    system,
                    "so3lr",
                    ROOT
                    / "viper_analysis"
                    / "sol3r"
                    / "azobenzene"
                    / f"{system}_2d"
                    / f"sol3r_azob_{system}_2d.bias",
                ),
                Record(
                    system,
                    "scratch",
                    ROOT / "raccoon_mace_scratch_azob" / "outputs" / f"mace_scratch_azob_{system}_2d.bias",
                ),
                Record(
                    system,
                    "ft_so3lr",
                    ROOT / "viper_ft_so3lr_azob" / "outputs" / f"ft_so3lr_azob_{system}_2d.bias",
                ),
                Record(
                    system,
                    "ft_mh1",
                    ROOT
                    / "raccoon_mace_ft_mh1_azob"
                    / "outputs"
                    / f"mace_ft_mh1_azob_{system}_2d.bias",
                ),
            ]
        )
    missing = [record.path for record in records if not record.path.exists()]
    if missing:
        raise FileNotFoundError("Missing bias logs:\n" + "\n".join(map(str, missing)))
    return records


def reconstruct(record: Record) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    run = parse_bias_log(record.path)
    n_hills = min(len(run.time_fs), int(np.searchsorted(run.time_fs, MAX_TIME_FS, side="right")))
    if n_hills == 0:
        raise ValueError(f"No hills available in {record.path}")
    grid1, grid2 = default_grid(d_cv1=1.0, d_cv2=1.0)
    fes = reconstruct_fes_2d(run, grid1, grid2, n_hills=n_hills)
    actual_time_fs = float(run.time_fs[n_hills - 1])
    np.savez_compressed(
        OUT_DIR / f"azobenzene_{record.system}_{record.model}_2d.npz",
        cv1_deg=grid1,
        cv2_deg=grid2,
        free_energy_eV=fes,
        actual_time_fs=actual_time_fs,
        n_hills=n_hills,
        source=str(record.path),
    )
    return grid1, grid2, fes, n_hills, actual_time_fs


def make_figure(
    system: str,
    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    vmax: float,
) -> None:
    fig = plt.figure(figsize=(15.0, 8.2), facecolor="white")
    panel_width = 0.185
    panel_height = 0.385
    top_y = 0.555
    bottom_y = 0.075
    top_x = (0.035, 0.255, 0.475, 0.695)
    bottom_x = top_x
    positions = tuple((x, top_y, panel_width, panel_height) for x in top_x) + tuple(
        (x, bottom_y, panel_width, panel_height) for x in bottom_x
    )
    axes = []
    contour = None
    levels = np.linspace(0.0, vmax, PLOT_LEVELS)

    for panel_index, (model, position) in enumerate(zip(MODELS, positions, strict=True)):
        ax = fig.add_axes(position)
        axes.append(ax)
        cv1, cv2, fes, actual_time_fs = surfaces[(system, model)]
        contour = ax.contourf(cv1, cv2, fes.T, levels=levels, cmap="RdBu_r", extend="max")
        ax.set_title(f"{LABELS[model]}\n{actual_time_fs / 1e6:.2f} ns", pad=6)
        ax.set_xlim(-180, 180)
        ax.set_ylim(60, 180)
        ax.set_xticks((-180, -90, 0, 90, 180))
        ax.set_yticks((60, 90, 120, 150, 180))
        ax.set_xlabel(r"CNNC dihedral, $\phi$ (deg)")
        if panel_index in (0, 4):
            ax.set_ylabel(r"CNN angle, $\theta$ (deg)")
        else:
            ax.set_ylabel("")
        ax.tick_params(direction="in", top=True, right=True, length=4.5, width=1.0)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5, width=0.8)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    colorbar_ax = fig.add_axes((0.925, bottom_y + 0.025, 0.018, panel_height - 0.05))
    cbar = fig.colorbar(contour, cax=colorbar_ax)
    cbar.set_label("Relative free energy (eV)")
    cbar.outline.set_linewidth(1.0)
    cbar.ax.tick_params(direction="out", length=4, width=0.9)
    for extension in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"azobenzene_{system}_2d_8model_paper.{extension}",
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

    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    summary: list[dict[str, str | int | float]] = []
    maxima = []
    for record in load_records():
        cv1, cv2, fes, n_hills, actual_time_fs = reconstruct(record)
        surfaces[(record.system, record.model)] = (cv1, cv2, fes, actual_time_fs)
        maxima.append(float(np.nanmax(fes)))
        summary.append(
            {
                "system": record.system,
                "model": record.model,
                "label": LABELS[record.model],
                "source": str(record.path),
                "n_hills": n_hills,
                "actual_time_fs": actual_time_fs,
                "actual_time_ns": actual_time_fs / 1e6,
            }
        )

    with (OUT_DIR / "azobenzene_2d_8model_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    # One common scale makes all panels directly comparable.
    vmax = float(np.ceil(max(maxima) * 10.0) / 10.0)
    for system in SYSTEMS:
        make_figure(system, surfaces, vmax)


if __name__ == "__main__":
    main()
