from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from azobenzene.scripts.barrier_extraction.fes_io import parse_bias_log
from azobenzene.scripts.barrier_extraction.fes_reconstruct import default_grid, reconstruct_fes_2d
from azobenzene.scripts.barrier_extraction.uncertainty import _pick_basin
from azobenzene.scripts.barrier_extraction.basins_mfep import enumerate_pathways

ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
AZOB_ANALYSIS = ROOT / "azobenzene" / "analysis"
VIPER_ANALYSIS = ROOT / "viper_analysis"
OUT_DIR = AZOB_ANALYSIS / "model_comparison"
RECON_DIR = OUT_DIR / "2ns_reconstructed"
MAX_TIME_FS = 2_000_000.0
PAPER_CNN_ANGLE_LIMITS = (60.0, 180.0)
EV_TO_KJMOL = 96.485
EV_TO_KCALMOL = 23.0605

MODELS = ("off", "mh1", "polar", "upet", "sol3r", "scratch", "ft_so3lr")
SYSTEMS = ("cis", "trans")
LABELS = {
    "off": "MACE-OFF",
    "omol": "MACE-OMOL",
    "mh1": "MACE-MH1",
    "polar": "MACE-Polar",
    "upet": "PET-SPICE",
    "sol3r": "SO3LR",
    "scratch": "MACE from scratch",
    "ft_so3lr": "Fine-tuned SO3LR",
}
@dataclass
class FESRecord:
    system: str
    model: str
    source: str
    path: Path


def find_pet_spice(system: str) -> Path:
    current = (
        Path("/work/gpuviper_ptmp/Enhanced_sampling/upet/azobenzene")
        / f"pet_spice_{system}_2d"
        / "outputs"
        / f"pet_spice_azob_{system}_2d.bias"
    )
    fallback = (
        VIPER_ANALYSIS
        / "upet"
        / "azobenzene"
        / f"pet_spice_{system}_2d"
        / f"pet_spice_azob_{system}_2d.bias"
    )
    return current if current.exists() else fallback


def load_records() -> list[FESRecord]:
    records: list[FESRecord] = []
    raccoon_jobs = {"off": "33975", "omol": "33976", "mh1": "33030", "polar": "33031"}
    for system in SYSTEMS:
        task = "0" if system == "cis" else "1"
        for model in ("off", "mh1", "polar"):
            records.append(
                FESRecord(
                    system=system,
                    model=model,
                    source="raccoon",
                    path=ROOT
                    / "azobenzene"
                    / "outputs"
                    / f"metad_azob_{system}_{model}_2d_raccoon_{model}_job{raccoon_jobs[model]}_task{task}_gpu1.txt",
                )
            )
        for model in ("upet", "sol3r"):
            if model == "upet":
                path = find_pet_spice(system)
            else:
                path = VIPER_ANALYSIS / model / "azobenzene" / f"{system}_2d" / f"{model}_azob_{system}_2d.bias"
            records.append(
                FESRecord(
                    system=system,
                    model=model,
                    source="viper",
                    path=path,
                )
            )
        records.extend(
            [
                FESRecord(
                    system=system,
                    model="scratch",
                    source="raccoon",
                    path=ROOT / "raccoon_mace_scratch_azob" / "outputs" / f"mace_scratch_azob_{system}_2d.bias",
                ),
                FESRecord(
                    system=system,
                    model="ft_so3lr",
                    source="viper",
                    path=ROOT / "viper_ft_so3lr_azob" / "outputs" / f"ft_so3lr_azob_{system}_2d.bias",
                ),
            ]
        )
    return [record for record in records if record.path.exists()]


def save_grid(path: Path, grid1: np.ndarray, grid2: np.ndarray, fes: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("cv1,cv2,free_energy\n")
        for j, cv2 in enumerate(grid2):
            for i, cv1 in enumerate(grid1):
                handle.write(f"{cv1:.8f},{cv2:.8f},{fes[i, j]:.12f}\n")


def extract_barriers(
    record: FESRecord,
    grid1: np.ndarray,
    grid2: np.ndarray,
    fes: np.ndarray,
    n_hills: int,
    actual_time_fs: float,
) -> dict[str, float | int | str]:
    cis_min = _pick_basin(fes, grid1, grid2, target=(0.0, 120.0))
    trans_min = _pick_basin(fes, grid1, grid2, target=(180.0, 120.0))
    paths = enumerate_pathways(fes, grid1, grid2, cis_min, trans_min)

    f_cis = float(fes[cis_min])
    f_trans = float(fes[trans_min])
    rot_saddle = float(paths["rotation"]["F_path"].max())
    inv_saddle = float(paths["inversion"]["F_path"].max())
    uncon_saddle = float(paths["unconstrained"]["F_path"].max())

    row = {
        "system": record.system,
        "model": record.model,
        "label": LABELS[record.model],
        "source": record.source,
        "bias_log": str(record.path),
        "n_hills": n_hills,
        "last_time_fs": actual_time_fs,
        "last_time_ns": actual_time_fs / 1e6,
        "cis_min_cv1_deg": float(grid1[cis_min[0]]),
        "cis_min_cv2_deg": float(grid2[cis_min[1]]),
        "trans_min_cv1_deg": float(grid1[trans_min[0]]),
        "trans_min_cv2_deg": float(grid2[trans_min[1]]),
        "F_cis_eV": f_cis,
        "F_trans_eV": f_trans,
        "dG_cis_minus_trans_eV": f_cis - f_trans,
        "rotation_cis_to_trans_eV": rot_saddle - f_cis,
        "rotation_trans_to_cis_eV": rot_saddle - f_trans,
        "inversion_cis_to_trans_eV": inv_saddle - f_cis,
        "inversion_trans_to_cis_eV": inv_saddle - f_trans,
        "unconstrained_cis_to_trans_eV": uncon_saddle - f_cis,
        "unconstrained_trans_to_cis_eV": uncon_saddle - f_trans,
    }
    for key, value in list(row.items()):
        if key.endswith("_eV"):
            row[f"{key[:-3]}_kJmol"] = float(value) * EV_TO_KJMOL
            row[f"{key[:-3]}_kcalmol"] = float(value) * EV_TO_KCALMOL
    return row


def reconstruct_grid(record: FESRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, dict[str, float | int | str]]:
    run = parse_bias_log(record.path)
    n_hills = int(np.searchsorted(run.time_fs, MAX_TIME_FS, side="right"))
    if n_hills <= 0:
        raise ValueError(f"{record.path}: no hills found before {MAX_TIME_FS:g} fs")

    grid1, grid2 = default_grid()
    fes = reconstruct_fes_2d(run, grid1, grid2, n_hills=n_hills)
    actual_time_fs = float(run.time_fs[n_hills - 1])

    output_csv = RECON_DIR / f"azobenzene_{record.system}_{record.model}_2d_{MAX_TIME_FS / 1e6:.1f}ns_reconstructed_fes.csv"
    save_grid(output_csv, grid1, grid2, fes)
    barriers = extract_barriers(record, grid1, grid2, fes, n_hills, actual_time_fs)
    return grid1, grid2, fes.T, n_hills, actual_time_fs, barriers


def panel_order(record: FESRecord) -> tuple[int, int]:
    model_index = MODELS.index(record.model)
    row = model_index // 3
    col = model_index % 3
    return row, col


def save_system_figure(
    system: str,
    records: list[FESRecord],
    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    vmin: float,
    vmax: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    ncols = 3
    nrows = int(np.ceil(len(MODELS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 4.1 * nrows), sharex=True, sharey=True, constrained_layout=True)
    contour_ref = None

    for record in records:
        row, col = panel_order(record)
        ax = axes[row, col]
        grid1, grid2, fes = surfaces[(record.system, record.model)]
        levels = np.linspace(vmin, vmax, 31)
        contour_ref = ax.contourf(grid1, grid2, fes, levels=levels, cmap="viridis", extend="max")
        ax.set_title(LABELS[record.model])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.15, linewidth=0.5)
        if col == 0:
            ax.set_ylabel("CNN angle (deg)")
        if row == nrows - 1:
            ax.set_xlabel("CNNC dihedral (deg)")

    for index in range(len(MODELS), axes.size):
        axes.flat[index].set_visible(False)

    cbar = fig.colorbar(contour_ref, ax=axes, shrink=0.98, pad=0.015)
    cbar.set_label("Relative free energy (arb. units)")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"azobenzene_{system}_2d_model_comparison.{ext}", dpi=300)
    plt.close(fig)


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
    RECON_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    if not records:
        raise SystemExit("No 2D bias logs found.")

    all_max = []
    all_cv1 = []
    all_cv2 = []
    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    summary_rows = ["system,model,source,bias_log,n_hills,last_time_fs,last_time_ns"]
    barrier_rows = []
    for record in records:
        grid1, grid2, fes, n_hills, actual_time_fs, barriers = reconstruct_grid(record)
        surfaces[(record.system, record.model)] = (grid1, grid2, fes)
        all_max.append(float(np.nanmax(fes)))
        all_cv1.extend([float(np.nanmin(grid1)), float(np.nanmax(grid1))])
        all_cv2.extend([float(np.nanmin(grid2)), float(np.nanmax(grid2))])
        summary_rows.append(
            f"{record.system},{record.model},{record.source},{record.path},{n_hills},{actual_time_fs:.2f},{actual_time_fs / 1e6:.6f}"
        )
        barrier_rows.append(barriers)

    (RECON_DIR / f"azobenzene_2d_{MAX_TIME_FS / 1e6:.1f}ns_reconstruction_summary.csv").write_text(
        "\n".join(summary_rows) + "\n"
    )
    barrier_header = list(barrier_rows[0])
    with (RECON_DIR / f"azobenzene_2d_{MAX_TIME_FS / 1e6:.1f}ns_barriers.csv").open("w") as handle:
        handle.write(",".join(barrier_header) + "\n")
        for row in barrier_rows:
            handle.write(",".join(str(row[column]) for column in barrier_header) + "\n")

    vmin = 0.0
    vmax = max(all_max)
    xlim = (min(all_cv1), max(all_cv1))
    ylim = PAPER_CNN_ANGLE_LIMITS

    for system in SYSTEMS:
        save_system_figure(
            system,
            [record for record in records if record.system == system],
            surfaces,
            vmin=vmin,
            vmax=vmax,
            xlim=xlim,
            ylim=ylim,
        )


if __name__ == "__main__":
    main()
