from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
OUT_DIR = ROOT / "analysis" / "publication_fes_comparison"


@dataclass(frozen=True)
class Curve:
    label: str
    path: Path
    color: str
    linestyle: str = "-"


SYSTEMS = {
    "malonaldehyd": {
        "title": "Malonaldehyde",
        "output": "malonaldehyde_fes_model_comparison",
        "xlim": (-1.05, 1.05),
        "curves": [
            Curve("DFT", ROOT / "analysis/orca_dft/malonaldehyd_8ps/umbrella_integration_orca_dft_8ps.csv", "black"),
            Curve("SO3LR", ROOT / "viper_analysis/sol3r/malonaldehyd/umbrella_integration_viper_sol3r.csv", "#0072B2"),
            Curve("MACE-OFF24 M", ROOT / "malonaldehyd/analysis/umbrella_integration_off24_medium.csv", "#009E73"),
            Curve("MACE-MH1", ROOT / "malonaldehyd/analysis/umbrella_integration_mh1_mh-1.csv", "#D55E00"),
            Curve("MACE-Polar M", ROOT / "malonaldehyd/analysis/umbrella_integration_polar_m.csv", "#CC79A7"),
            Curve("PET-SPICE", ROOT / "viper_analysis/upet/malonaldehyd/umbrella_integration_viper_upet_pet_spice.csv", "#E69F00"),
            Curve(
                "PET-SPICE_rot",
                ROOT
                / "viper_analysis/upet/malonaldehyd_pet_spice_rotavg3_43w/umbrella_integration_viper_upet_pet_spice_rot.csv",
                "#ff7f00",
            ),
        ],
    },
    "f-malonaldehyd": {
        "title": "F-malonaldehyde",
        "output": "f_malonaldehyde_fes_model_comparison",
        "xlim": (-1.05, 1.25),
        "curves": [
            Curve(
                "DFT",
                ROOT
                / "analysis/orca_dft/f-malonaldehyd_4p5ps_mlip_cv"
                / "umbrella_integration_orca_dft_4p5ps_mlip_cv.csv",
                "black",
            ),
            Curve("SO3LR", ROOT / "viper_analysis/sol3r/f-malonaldehyd/umbrella_integration_viper_sol3r.csv", "#0072B2"),
            Curve("MACE-OFF24 M", ROOT / "f-malonaldehyd/analysis/umbrella_integration_off24_medium.csv", "#009E73"),
            Curve("MACE-MH1", ROOT / "f-malonaldehyd/analysis/umbrella_integration_mh1_mh-1.csv", "#D55E00"),
            Curve("MACE-Polar M", ROOT / "f-malonaldehyd/analysis/umbrella_integration_polar_m.csv", "#CC79A7"),
            Curve("PET-SPICE", ROOT / "viper_analysis/upet/f-malonaldehyd/umbrella_integration_viper_upet_pet_spice.csv", "#E69F00"),
            Curve(
                "PET-SPICE_rot",
                ROOT
                / "viper_analysis/upet/f-malonaldehyd_pet_spice_rotavg3_43w/umbrella_integration_viper_upet_pet_spice_rot.csv",
                "#ff7f00",
            ),
        ],
    },
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def read_pmf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names or ()
    x = np.atleast_1d(data["mean_cv"]).astype(float)
    column = "free_energy_diff_gp" if "free_energy_diff_gp" in names else "free_energy_ui"
    y = np.atleast_1d(data[column]).astype(float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    order = np.argsort(x)
    return x[order], y[order] - np.min(y)


def draw(ax: plt.Axes, config: dict, show_legend: bool = True) -> None:
    curves = config["curves"]
    loaded = [(curve, *read_pmf(curve.path)) for curve in curves]
    x_min, x_max = config.get("xlim", (None, None))
    if x_min is None or x_max is None:
        x_min = max(float(np.min(x)) for _, x, _ in loaded)
        x_max = min(float(np.max(x)) for _, x, _ in loaded)

    dft, x, y = loaded[0]
    mask = (x >= x_min) & (x <= x_max)
    ax.plot(x[mask], y[mask], color=dft.color, lw=3.0, alpha=1.0, label=dft.label, zorder=2)

    for curve, x, y in loaded[1:]:
        mask = (x >= x_min) & (x <= x_max)
        ax.plot(
            x[mask],
            y[mask],
            color=curve.color,
            linestyle=curve.linestyle,
            lw=3.0,
            alpha=0.6,
            label=curve.label,
            zorder=2,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Proton-transfer coordinate / $\mathrm{\AA}$")
    ax.set_ylabel(r"Free energy / eV")
    ax.set_title(config["title"], fontweight="semibold", pad=10)
    ax.grid(color="#d8d8d8", linewidth=0.7, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    if show_legend:
        ax.legend(
            frameon=False,
            ncol=1,
            loc="upper right",
            handlelength=3.0,
        )


def main() -> None:
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for config in SYSTEMS.values():
        fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
        draw(ax, config)
        for ext in ("png", "pdf"):
            path = OUT_DIR / f"{config['output']}.{ext}"
            fig.savefig(path, dpi=400)
            print(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    for ax, config in zip(axes, SYSTEMS.values()):
        draw(ax, config, show_legend=True)
    axes[1].set_ylabel("")
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"malonaldehyde_fes_model_comparison_combined.{ext}"
        fig.savefig(path, dpi=400)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
