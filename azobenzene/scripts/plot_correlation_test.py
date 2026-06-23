#!/usr/bin/env python3
"""Cross-correlation plots (energy + forces) for all trained azobenzene models.

Reads NPZ prediction files from test_predictions/ and generates a
2 × N_models figure (top = energy, bottom = forces) in publication style.

Energy RMSE: per-atom, single-offset subtracted (mean DFT_pa - pred_pa).
Force RMSE:  component-wise over all atoms (no offset).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ngoen")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PRED_ROOT = Path(
    "/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/"
    "azobenzene_dft_benchmark/results/split_predictions"
)
OUT_DIR = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/analysis/publication_fes_comparison")
SPLITS = ("train", "valid", "test")

# E0s (wB97M-V/def2-TZVPD, eV) — used to convert raw MACE DFT reference
# to the same shifted frame as SO3LR for per-atom energy plotting
E0S = {"H": -13.445423, "C": -1029.854265, "N": -1485.541877}

MODEL_CONFIG: list[tuple[str, str, str]] = [
    ("scratch",         "MACE\nScratch",   "#E69F00"),
    ("ft_off24",        "MACE\nFT-OFF24",  "#009E73"),
    ("ft_mh1",          "MACE\nFT-MH1",   "#D55E00"),
    ("so3lr_ft",        "SO3LR FT",        "#0072B2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_dft_reference(pred_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw DFT energies, forces, and per-structure n_atoms from MACE test set."""
    npz = np.load(pred_dir / "dft_reference.npz")
    return npz["e"], npz["forces"], npz["n_atoms"]


def load_mace_predictions(name: str, pred_dir: Path) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    path = pred_dir / f"predictions_{name}.npz"
    if not path.exists():
        return None
    npz = np.load(path)
    return npz["e"], npz["forces"], float(npz["rmse_e_mev"]), float(npz["rmse_f_evang"])


def load_so3lr_predictions(pred_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    """Returns (e_dft_so3lr, f_dft_so3lr, e_pred, f_pred, rmse_e, rmse_f) or None."""
    path = pred_dir / "predictions_so3lr_ft.npz"
    if not path.exists():
        return None
    npz = np.load(path)
    return (
        npz["e_dft"], npz["f_dft"],
        npz["e"], npz["forces"],
        float(npz["rmse_e_mev"]), float(npz["rmse_f_evang"]),
    )


def centered_per_atom(e: np.ndarray, n_atoms: np.ndarray) -> np.ndarray:
    """Per-atom energy, mean-centered. Returns array of shape (N,)."""
    pa = e / n_atoms
    return (pa - pa.mean()) * 1000  # meV/atom


def panel_energy(
    ax: plt.Axes,
    e_dft: np.ndarray,
    e_pred: np.ndarray,
    n_atoms: np.ndarray,
    rmse_e: float,
    color: str,
    title: str,
    limits: tuple[float, float],
) -> None:
    x = centered_per_atom(e_dft, n_atoms)
    y = centered_per_atom(e_pred, n_atoms)

    ax.scatter(x, y, s=8, color=color, alpha=0.6, linewidths=0, rasterized=True)

    lower, upper = limits
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    # y = x reference line
    ax.plot([lower, upper], [lower, upper], "k--", lw=0.8, alpha=0.5)

    ax.set_xlabel("DFT (meV/atom)")
    ax.set_ylabel("Predicted (meV/atom)")
    ax.set_title(title, fontweight="semibold", pad=6)

    ax.text(
        0.97, 0.05,
        f"RMSE = {rmse_e:.1f} meV/atom",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
    )

    loc = ticker.MaxNLocator(nbins=5)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))


def panel_forces(
    ax: plt.Axes,
    f_dft: np.ndarray,
    f_pred: np.ndarray,
    rmse_f: float,
    color: str,
) -> None:
    # Flatten to 1D components; subsample for speed if large
    fx = f_dft.ravel()
    fy = f_pred.ravel()
    if len(fx) > 30_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(fx), 30_000, replace=False)
        fx, fy = fx[idx], fy[idx]

    ax.scatter(fx, fy, s=4, color=color, alpha=0.4, linewidths=0, rasterized=True)

    lim_val = max(abs(fx).max(), abs(fy).max()) * 1.08
    ax.set_xlim(-lim_val, lim_val)
    ax.set_ylim(-lim_val, lim_val)
    ax.plot([-lim_val, lim_val], [-lim_val, lim_val], "k--", lw=0.8, alpha=0.5)

    ax.set_xlabel("DFT force (eV/Å)")
    ax.set_ylabel("Predicted force (eV/Å)")

    ax.text(
        0.97, 0.05,
        f"RMSE = {rmse_f:.3f} eV/Å",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
    )

    loc = ticker.MaxNLocator(nbins=5)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))


def plot_split(split: str, pred_dir: Path) -> list[dict[str, str | float]]:
    e_dft_raw, f_dft_raw, n_atoms = load_dft_reference(pred_dir)
    # Panels to plot: collect (e_dft, f_dft, e_pred, f_pred, rmse_e, rmse_f, label, color)
    panels = []
    summary = []
    for key, label, color in MODEL_CONFIG:
        if key == "so3lr_ft":
            result = load_so3lr_predictions(pred_dir)
            if result is None:
                print(f"SKIP {split}/{key}: predictions file not found")
                continue
            e_dft_so3lr, f_dft_so3lr, e_pred, f_pred, rmse_e, rmse_f = result
            panels.append((e_dft_so3lr, f_dft_so3lr, e_pred, f_pred, rmse_e, rmse_f, label, color, n_atoms))
        else:
            result = load_mace_predictions(key, pred_dir)
            if result is None:
                print(f"SKIP {split}/{key}: predictions file not found")
                continue
            e_pred, f_pred, rmse_e, rmse_f = result
            panels.append((e_dft_raw, f_dft_raw, e_pred, f_pred, rmse_e, rmse_f, label, color, n_atoms))
        summary.append(
            {
                "split": split,
                "model": key,
                "label": label.replace("\n", " "),
                "energy_rmse_meV_per_atom": rmse_e,
                "force_rmse_eV_per_A": rmse_f,
            }
        )

    if not panels:
        raise FileNotFoundError("No prediction files found. Run the prediction scripts first.")

    energy_values = []
    for e_dft, _, e_pred, _, _, _, _, _, n_at in panels:
        energy_values.extend((centered_per_atom(e_dft, n_at), centered_per_atom(e_pred, n_at)))
    global_min = min(float(values.min()) for values in energy_values)
    global_max = max(float(values.max()) for values in energy_values)
    energy_limits = (
        2.0 * global_min if global_min < 0.0 else 0.5 * global_min,
        1.1 * global_max if global_max > 0.0 else 0.9 * global_max,
    )

    n_models = len(panels)
    fig, axes = plt.subplots(
        2, n_models,
        figsize=(3.5 * n_models, 7.0),
        constrained_layout=True,
    )
    if n_models == 1:
        axes = axes.reshape(2, 1)

    for col, (e_dft, f_dft, e_pred, f_pred, rmse_e, rmse_f, label, color, n_at) in enumerate(panels):
        panel_energy(axes[0, col], e_dft, e_pred, n_at, rmse_e, color, label, energy_limits)
        panel_forces(axes[1, col], f_dft, f_pred, rmse_f, color)

    # Row labels
    axes[0, 0].set_ylabel("Predicted energy (meV/atom)", fontsize=12)
    axes[1, 0].set_ylabel("Predicted force (eV/Å)", fontsize=12)

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"azob_{split}_correlation.{ext}"
        fig.savefig(out, dpi=400)
        print(f"Saved {out}")
    plt.close(fig)
    return summary


def main() -> None:
    args = parse_args()
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    requested_splits = SPLITS if args.split == "all" else (args.split,)
    for split in requested_splits:
        pred_dir = PRED_ROOT / split
        if not (pred_dir / "dft_reference.npz").exists():
            print(f"SKIP {split}: missing {pred_dir / 'dft_reference.npz'}")
            continue
        summary.extend(plot_split(split, pred_dir))

    if not summary:
        raise FileNotFoundError(f"No completed split predictions found under {PRED_ROOT}")
    with (OUT_DIR / "azob_split_rmse_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
