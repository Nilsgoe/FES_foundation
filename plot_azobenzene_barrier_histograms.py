from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
OUT_DIR = ROOT / "azobenzene" / "analysis" / "model_comparison"
INPUT_CSV = OUT_DIR / "2ns_reconstructed" / "azobenzene_2d_2.0ns_barriers.csv"
SUMMARY_CSV = OUT_DIR / "2ns_reconstructed" / "azobenzene_2d_2.0ns_barrier_histogram_summary.csv"

MODEL_ORDER = ["off", "mh1", "polar", "upet", "sol3r", "scratch", "ft_so3lr"]
LABEL_OVERRIDES = {
    "off": "MACE-OFF",
    "mh1": "MACE-MH1",
    "polar": "MACE-Polar",
    "upet": "PET-SPICE",
    "sol3r": "SO3LR",
    "scratch": "MACE from scratch",
    "ft_so3lr": "Fine-tuned SO3LR",
}
COLORS = {
    "off": "#009E73",
    "mh1": "#D55E00",
    "polar": "#CC79A7",
    "upet": "#E69F00",
    "sol3r": "#0072B2",
    "scratch": "#56B4E9",
    "ft_so3lr": "#003B73",
}
MARKERS = {
    "off": "o",
    "mh1": "P",
    "polar": "X",
    "upet": "v",
    "sol3r": "D",
    "scratch": "s",
    "ft_so3lr": "^",
}


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def load_summary() -> list[dict[str, float | str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with INPUT_CSV.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[row["model"]].append(row)

    rows: list[dict[str, float | str]] = []
    for model in MODEL_ORDER:
        entries = grouped.get(model, [])
        if not entries:
            continue
        c2t = [float(row["unconstrained_cis_to_trans_eV"]) for row in entries]
        t2c = [float(row["unconstrained_trans_to_cis_eV"]) for row in entries]
        dg = [float(row["dG_cis_minus_trans_eV"]) for row in entries]
        c2t_mean, c2t_std = mean_std(c2t)
        t2c_mean, t2c_std = mean_std(t2c)
        dg_mean, dg_std = mean_std(dg)
        rows.append(
            {
                "model": model,
                "label": LABEL_OVERRIDES.get(model, entries[0]["label"]),
                "n_replicates": len(entries),
                "cis_to_trans_barrier_eV_mean": c2t_mean,
                "cis_to_trans_barrier_eV_std": c2t_std,
                "trans_to_cis_barrier_eV_mean": t2c_mean,
                "trans_to_cis_barrier_eV_std": t2c_std,
                "deltaG_cis_minus_trans_eV_mean": dg_mean,
                "deltaG_cis_minus_trans_eV_std": dg_std,
            }
        )
    return rows


def write_summary(rows: list[dict[str, float | str]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | str]]) -> None:
    labels = [str(row["label"]) for row in rows]
    models = [str(row["model"]) for row in rows]
    x = np.arange(len(rows), dtype=float)
    width = 0.36

    c2t = np.asarray([float(row["cis_to_trans_barrier_eV_mean"]) for row in rows])
    c2t_err = np.asarray([float(row["cis_to_trans_barrier_eV_std"]) for row in rows])
    t2c = np.asarray([float(row["trans_to_cis_barrier_eV_mean"]) for row in rows])
    t2c_err = np.asarray([float(row["trans_to_cis_barrier_eV_std"]) for row in rows])
    dg = np.asarray([float(row["deltaG_cis_minus_trans_eV_mean"]) for row in rows])
    dg_err = np.asarray([float(row["deltaG_cis_minus_trans_eV_std"]) for row in rows])
    colors = [COLORS.get(model, "#4a5568") for model in models]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 1.2,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "savefig.bbox": "tight",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)

    ax = axes[0]
    markers = [MARKERS[model] for model in models]
    for i, (label, color, marker) in enumerate(zip(labels, colors, markers, strict=True)):
        ax.errorbar(
            c2t[i],
            t2c[i],
            xerr=c2t_err[i],
            yerr=t2c_err[i],
            fmt=marker,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=9.5,
            elinewidth=1.1,
            capsize=3,
            label=label,
        )
    ax.set_xlabel(r"cis $\rightarrow$ trans barrier / eV")
    ax.set_ylabel(r"trans $\rightarrow$ cis barrier / eV")
    ax.grid(alpha=0.22, linewidth=0.8)
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.bar(
        x,
        dg,
        0.58,
        yerr=dg_err,
        color=colors,
        alpha=0.82,
        edgecolor="black",
        linewidth=0.8,
        capsize=3,
    )
    ax.set_ylabel(r"$\Delta G_{\mathrm{cis-trans}}$ / eV")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.text(
        0.015,
        0.95,
        r"positive: cis less stable than trans",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
    )

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"azobenzene_2d_2ns_barrier_histograms.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    rows = load_summary()
    if not rows:
        raise SystemExit(f"No rows found in {INPUT_CSV}")
    write_summary(rows)
    plot(rows)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {OUT_DIR / 'azobenzene_2d_2ns_barrier_histograms.png'}")


if __name__ == "__main__":
    main()
