from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
INPUT_CSV = ROOT / "analysis/barrier_overview_malonaldehyd_models.csv"
OUT_DIR = ROOT / "analysis/publication_fes_comparison"

# Exactly the same colors and model set as plot_malon_publication_fes_comparison.py
MODEL_CONFIG: dict[str, tuple[str, str, str]] = {
    "wB97M-V_def2-TZVPD": ("DFT",         "black",   "*"),
    "SO3LR":               ("SO3LR",        "#0072B2", "D"),
    "off24_medium":        ("MACE-OFF24 M", "#009E73", "o"),
    "mh1_mh-1":            ("MACE-MH1",     "#D55E00", "s"),
    "polar_m":             ("MACE-Polar M", "#CC79A7", "^"),
    "upet_pet-spice-l":    ("PET-SPICE",    "#E69F00", "P"),
    "upet_pet-spice-l_rot": ("PET-SPICE_rot", "#ff7f00", "X"),
}


def clean_row(row: dict[str, str]) -> dict[str, str]:
    return {k.strip(): v.strip() for k, v in row.items()}


def read_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open(newline="") as fh:
        rows = [clean_row(r) for r in csv.DictReader(fh)]
    return [r for r in rows if r.get("model") in MODEL_CONFIG]


def as_float(row: dict[str, str], key: str) -> float | None:
    v = row.get(key, "").strip()
    try:
        return float(v) if v else None
    except ValueError:
        return None


def setup_style() -> None:
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


def scatter_panel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    prefix: str,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    for row in rows:
        model = row["model"]
        x = as_float(row, f"{prefix}_{x_key}")
        y = as_float(row, f"{prefix}_{y_key}")
        if x is None or y is None:
            continue
        label, color, marker = MODEL_CONFIG[model]
        is_dft = model.startswith("wB97M")
        ax.scatter(
            x, y,
            s=200 if is_dft else 80,
            color=color,
            marker=marker,
            alpha=0.95,
            edgecolor="black" if is_dft else "white",
            linewidth=1.5 if is_dft else 0.7,
            zorder=6 if is_dft else 3,
        )

    loc = ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
    fmt = ticker.FormatStrFormatter("%.2f")
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="semibold", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


def build_legend_handles() -> tuple[list, list[str]]:
    handles, labels = [], []
    for model, (label, color, marker) in MODEL_CONFIG.items():
        is_dft = model.startswith("wB97M")
        handles.append(
            plt.Line2D(
                [0], [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="black" if is_dft else "white",
                markeredgewidth=1.4 if is_dft else 0.7,
                markersize=11 if is_dft else 8,
            )
        )
        labels.append(label)
    return handles, labels


def make_combined_2x2(rows: list[dict[str, str]]) -> None:
    """2×2: top row = barrier heights, bottom row = minima; left = malon, right = f-malon."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.0), constrained_layout=True)

    scatter_panel(
        axes[0, 0], rows, "malon",
        "left_barrier_height", "right_barrier_height",
        "Left barrier height (eV)", "Right barrier height (eV)",
        "Malonaldehyde",
    )
    scatter_panel(
        axes[0, 1], rows, "f_malon",
        "left_barrier_height", "right_barrier_height",
        "Left barrier height (eV)", "Right barrier height (eV)",
        "F-malonaldehyde",
    )
    scatter_panel(
        axes[1, 0], rows, "malon",
        "left_min_cv", "right_min_cv",
        r"Left minimum / $\mathrm{\AA}$", r"Right minimum / $\mathrm{\AA}$",
        "Malonaldehyde",
    )
    scatter_panel(
        axes[1, 1], rows, "f_malon",
        "left_min_cv", "right_min_cv",
        r"Left minimum / $\mathrm{\AA}$", r"Right minimum / $\mathrm{\AA}$",
        "F-malonaldehyde",
    )


    handles, labels = build_legend_handles()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=6, frameon=False,
        bbox_to_anchor=(0.5, -0.055),
    )
    _save(fig, "malon_fmalon_scatter_combined")


def make_barrier_combined(rows: list[dict[str, str]]) -> None:
    """1×2: barrier heights for malon and f-malon side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), constrained_layout=True)

    scatter_panel(
        axes[0], rows, "malon",
        "left_barrier_height", "right_barrier_height",
        "Left barrier height (eV)", "Right barrier height (eV)",
        "Malonaldehyde",
    )
    scatter_panel(
        axes[1], rows, "f_malon",
        "left_barrier_height", "right_barrier_height",
        "Left barrier height (eV)", "Right barrier height (eV)",
        "F-malonaldehyde",
    )

    handles, labels = build_legend_handles()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=6, frameon=False,
        bbox_to_anchor=(0.5, -0.09),
    )
    _save(fig, "malon_fmalon_barrier_scatter")


def make_minima_combined(rows: list[dict[str, str]]) -> None:
    """1×2: minima positions for malon and f-malon side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), constrained_layout=True)

    scatter_panel(
        axes[0], rows, "malon",
        "left_min_cv", "right_min_cv",
        r"Left minimum / $\mathrm{\AA}$", r"Right minimum / $\mathrm{\AA}$",
        "Malonaldehyde",
    )
    scatter_panel(
        axes[1], rows, "f_malon",
        "left_min_cv", "right_min_cv",
        r"Left minimum / $\mathrm{\AA}$", r"Right minimum / $\mathrm{\AA}$",
        "F-malonaldehyde",
    )

    handles, labels = build_legend_handles()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=6, frameon=False,
        bbox_to_anchor=(0.5, -0.09),
    )
    _save(fig, "malon_fmalon_minima_scatter")


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=400)
        print(out)
    plt.close(fig)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    make_combined_2x2(rows)
    make_barrier_combined(rows)
    make_minima_combined(rows)


if __name__ == "__main__":
    main()
