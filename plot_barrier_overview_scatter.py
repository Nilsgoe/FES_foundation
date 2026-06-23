from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
INPUT_CSV = ROOT / "analysis/barrier_overview_malonaldehyd_models.csv"
OUT_DIR = ROOT / "analysis"

FAMILY_STYLE = {
    "UPET": ("#d95f02", "s"),
    "SO3LR": ("#7570b3", "D"),
    "DFT": ("#111111", "*"),
}

MODEL_STYLE = {
    "off24_medium": ("#1b9e77", "o"),
    "omol_extra_large": ("#66a61e", "^"),
    "mh1_mh-1": ("#008b8b", "P"),
    "polar_m": ("#2b6cb0", "X"),
    "upet_pet-oam-xl": ("#d95f02", "s"),
    "upet_pet-spice-l": ("#e66101", "v"),
    "SO3LR": ("#7570b3", "D"),
    "wB97M-V_def2-TZVPD": ("#111111", "*"),
}

INCLUDED_MACE_MODELS = {
    "off24_medium",
    "omol_extra_large",
    "mh1_mh-1",
    "polar_m",
}


def clean_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


def read_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open(newline="") as handle:
        rows = [clean_row(row) for row in csv.DictReader(handle)]
    return [row for row in rows if include_row(row)]


def include_row(row: dict[str, str]) -> bool:
    if row.get("family") != "MACE":
        return True
    return row.get("model") in INCLUDED_MACE_MODELS


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def model_label(row: dict[str, str]) -> str:
    family = row["family"]
    model = row["model"]
    if family == "MACE":
        mapping = {
            "off24_medium": "MACE-OFF24 M",
            "omol_extra_large": "MACE-OMOL XL",
            "mh1_mh-1": "MACE-MH1",
            "polar_m": "MACE-Polar M",
        }
        return mapping.get(model, model.replace("_", " "))
    if family == "DFT":
        return "DFT"
    if family == "UPET" and "pet-spice" in model:
        return "PET-SPICE"
    if family == "UPET":
        return "UPET"
    return model


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def scatter_panel(
    ax,
    rows: list[dict[str, str]],
    system_prefix: str,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    for row in rows:
        x = as_float(row, f"{system_prefix}_{x_key}")
        y = as_float(row, f"{system_prefix}_{y_key}")
        if x is None or y is None:
            continue
        family = row["family"]
        color, marker = MODEL_STYLE.get(row["model"], FAMILY_STYLE.get(family, ("#666666", "o")))
        size = 210 if family == "DFT" else 78
        edgecolor = "#ffcc00" if family == "DFT" else "white"
        linewidth = 1.6 if family == "DFT" else 0.7
        zorder = 6 if family == "DFT" else 3
        ax.scatter(
            x,
            y,
            s=size,
            color=color,
            marker=marker,
            alpha=0.95,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.24)


def add_family_legend(fig) -> None:
    handles = []
    labels = []
    for model, (color, marker) in MODEL_STYLE.items():
        label_row = {"family": "MACE" if model in INCLUDED_MACE_MODELS else "", "model": model}
        if model.startswith("upet"):
            label_row["family"] = "UPET"
        elif model == "SO3LR":
            label_row["family"] = "SO3LR"
        elif model.startswith("wB97M"):
            label_row["family"] = "DFT"
        marker_size = 11 if label_row["family"] == "DFT" else 8
        edgecolor = "#ffcc00" if label_row["family"] == "DFT" else "white"
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor=edgecolor,
                markeredgewidth=1.4 if label_row["family"] == "DFT" else 0.7,
                markersize=marker_size,
            )
        )
        labels.append(model_label(label_row))
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.09))


def make_barrier_plot(rows: list[dict[str, str]]) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    scatter_panel(
        axes[0],
        rows,
        "malon",
        "left_barrier_height",
        "right_barrier_height",
        "Left barrier height (eV)",
        "Right barrier height (eV)",
        "Malonaldehyde",
    )
    scatter_panel(
        axes[1],
        rows,
        "f_malon",
        "left_barrier_height",
        "right_barrier_height",
        "Left barrier height (eV)",
        "Right barrier height (eV)",
        "F-malonaldehyde",
    )
    fig.suptitle("Barrier Heights From Fitted Umbrella PMFs", fontsize=14, fontweight="bold")
    add_family_legend(fig)
    outputs = []
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"barrier_height_left_vs_right.{ext}"
        fig.savefig(out, dpi=300)
        outputs.append(out)
    plt.close(fig)
    return outputs


def make_minima_plot(rows: list[dict[str, str]]) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    scatter_panel(
        axes[0],
        rows,
        "malon",
        "left_min_cv",
        "right_min_cv",
        "Left minimum CV (A)",
        "Right minimum CV (A)",
        "Malonaldehyde",
    )
    scatter_panel(
        axes[1],
        rows,
        "f_malon",
        "left_min_cv",
        "right_min_cv",
        "Left minimum CV (A)",
        "Right minimum CV (A)",
        "F-malonaldehyde",
    )
    fig.suptitle("Minimum Positions From Fitted Umbrella PMFs", fontsize=14, fontweight="bold")
    add_family_legend(fig)
    outputs = []
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"minimum_position_left_vs_right.{ext}"
        fig.savefig(out, dpi=300)
        outputs.append(out)
    plt.close(fig)
    return outputs


def make_f_malon_only_plots(rows: list[dict[str, str]]) -> list[Path]:
    outputs = []
    plots = (
        (
            "left_barrier_height",
            "right_barrier_height",
            "Left barrier height (eV)",
            "Right barrier height (eV)",
            "f_malonaldehyd_barrier_height_left_vs_right",
        ),
        (
            "left_min_cv",
            "right_min_cv",
            "Left minimum CV (A)",
            "Right minimum CV (A)",
            "f_malonaldehyd_minimum_position_left_vs_right",
        ),
    )
    for x_key, y_key, xlabel, ylabel, stem in plots:
        fig, ax = plt.subplots(figsize=(6.4, 5.5), constrained_layout=True)
        scatter_panel(ax, rows, "f_malon", x_key, y_key, xlabel, ylabel, "F-malonaldehyde")
        add_family_legend(fig)
        for ext in ("png", "pdf"):
            out = OUT_DIR / f"{stem}.{ext}"
            fig.savefig(out, dpi=300)
            outputs.append(out)
        plt.close(fig)
    return outputs


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    outputs = [*make_barrier_plot(rows), *make_minima_plot(rows), *make_f_malon_only_plots(rows)]
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
