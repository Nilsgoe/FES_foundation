#!/usr/bin/env python3
"""Plot test-set DFT correlations for pretrained and fine-tuned SO3LR."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ngoen")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
PRED_DIR = ROOT / "azobenzene_dft_benchmark/results/split_predictions/test"
OUT_DIR = ROOT / "analysis/publication_fes_comparison"
MODELS = (
    ("so3lr_pretrained", "SO3LR", "#56B4E9"),
    ("so3lr_ft", "Fine-tuned SO3LR", "#0072B2"),
)


def centered_per_atom(values: np.ndarray, n_atoms: np.ndarray) -> np.ndarray:
    per_atom = values / n_atoms
    return (per_atom - per_atom.mean()) * 1000.0


def load(key: str) -> dict[str, np.ndarray | float]:
    data = np.load(PRED_DIR / f"predictions_{key}.npz")
    return {name: data[name] for name in data.files}


def padded_limits(arrays: list[np.ndarray], lower_factor: float = 1.1) -> tuple[float, float]:
    minimum = min(float(array.min()) for array in arrays)
    maximum = max(float(array.max()) for array in arrays)
    return lower_factor * minimum, 1.1 * maximum


def main() -> None:
    loaded = [(key, label, color, load(key)) for key, label, color in MODELS]
    energy_arrays = []
    force_arrays = []
    for _, _, _, data in loaded:
        n_atoms = np.asarray(data["n_atoms"])
        energy_arrays.extend(
            [
                centered_per_atom(np.asarray(data["e_dft"]), n_atoms),
                centered_per_atom(np.asarray(data["e"]), n_atoms),
            ]
        )
        force_arrays.extend([np.asarray(data["f_dft"]).ravel(), np.asarray(data["forces"]).ravel()])

    energy_limits = padded_limits(energy_arrays, lower_factor=2.0)
    force_limit = 1.1 * max(float(np.abs(array).max()) for array in force_arrays)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.linewidth": 1.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.0), constrained_layout=True)
    summary = []
    rng = np.random.default_rng(42)
    for column, (key, label, color, data) in enumerate(loaded):
        n_atoms = np.asarray(data["n_atoms"])
        x_energy = centered_per_atom(np.asarray(data["e_dft"]), n_atoms)
        y_energy = centered_per_atom(np.asarray(data["e"]), n_atoms)
        axes[0, column].scatter(x_energy, y_energy, s=10, color=color, alpha=0.65, linewidths=0)
        axes[0, column].plot(energy_limits, energy_limits, "k--", lw=0.9, alpha=0.55)
        axes[0, column].set(xlim=energy_limits, ylim=energy_limits, title=label)
        axes[0, column].set_xlabel("DFT energy (meV/atom)")
        axes[0, column].set_ylabel("Predicted energy (meV/atom)")
        axes[0, column].text(
            0.97,
            0.05,
            f"RMSE = {float(data['rmse_e_mev']):.1f} meV/atom",
            transform=axes[0, column].transAxes,
            ha="right",
        )

        x_force = np.asarray(data["f_dft"]).ravel()
        y_force = np.asarray(data["forces"]).ravel()
        if x_force.size > 30_000:
            indices = rng.choice(x_force.size, 30_000, replace=False)
            x_force, y_force = x_force[indices], y_force[indices]
        axes[1, column].scatter(x_force, y_force, s=4, color=color, alpha=0.4, linewidths=0, rasterized=True)
        axes[1, column].plot((-force_limit, force_limit), (-force_limit, force_limit), "k--", lw=0.9, alpha=0.55)
        axes[1, column].set(xlim=(-force_limit, force_limit), ylim=(-force_limit, force_limit))
        axes[1, column].set_xlabel(r"DFT force (eV/$\mathrm{\AA}$)")
        axes[1, column].set_ylabel(r"Predicted force (eV/$\mathrm{\AA}$)")
        axes[1, column].text(
            0.97,
            0.05,
            rf"RMSE = {float(data['rmse_f_evang']):.3f} eV/$\mathrm{{\AA}}$",
            transform=axes[1, column].transAxes,
            ha="right",
        )
        summary.append(
            {
                "model": key,
                "label": label,
                "energy_rmse_meV_per_atom": float(data["rmse_e_mev"]),
                "force_rmse_eV_per_A": float(data["rmse_f_evang"]),
            }
        )

    for axis in axes.ravel():
        axis.tick_params(direction="in", top=True, right=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"azob_so3lr_pretrained_vs_finetuned_correlation.{extension}", dpi=400)
    plt.close(fig)
    with (OUT_DIR / "azob_so3lr_pretrained_vs_finetuned_rmse.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
