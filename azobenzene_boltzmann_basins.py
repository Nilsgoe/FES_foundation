from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
IN_DIR = ROOT / "azobenzene" / "analysis" / "model_comparison" / "2ns_8model_paper"
OUT_DIR = ROOT / "azobenzene" / "analysis" / "model_comparison" / "2ns_8model_boltzmann_basins"
KB_EV_PER_K = 8.617333262145e-5
EV_TO_KJMOL = 96.48533212
EV_TO_KCALMOL = 23.06054887
TEMPERATURE_K = 300.0
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
PLOT_LEVELS = 13


def load_surface(system: str, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    path = IN_DIR / f"azobenzene_{system}_{model}_2d.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    return (
        np.asarray(data["cv1_deg"], dtype=float),
        np.asarray(data["cv2_deg"], dtype=float),
        np.asarray(data["free_energy_eV"], dtype=float),
        float(data["actual_time_fs"]) / 1e6,
    )


def grid_widths(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.ones_like(values)
    widths = np.empty_like(values)
    widths[1:-1] = 0.5 * (values[2:] - values[:-2])
    widths[0] = 0.5 * (values[1] - values[0])
    widths[-1] = 0.5 * (values[-1] - values[-2])
    return np.clip(np.abs(widths), np.finfo(float).tiny, None)


def boltzmann_partition(F: np.ndarray, cv1: np.ndarray, cv2: np.ndarray, mask: np.ndarray) -> float:
    beta = 1.0 / (KB_EV_PER_K * TEMPERATURE_K)
    w1 = grid_widths(cv1)[:, None]
    w2 = grid_widths(cv2)[None, :]
    shifted = F - np.nanmin(F)
    weights = w1 * w2
    return float(np.nansum(weights[mask] * np.exp(-beta * shifted[mask])))


def marginal_phi_free_energy(cv1: np.ndarray, cv2: np.ndarray, F: np.ndarray) -> np.ndarray:
    beta = 1.0 / (KB_EV_PER_K * TEMPERATURE_K)
    w2 = grid_widths(cv2)[None, :]
    shifted = F - np.nanmin(F)
    z_phi = np.nansum(w2 * np.exp(-beta * shifted), axis=1)
    g = -np.log(np.clip(z_phi, np.finfo(float).tiny, None)) / beta
    return g - np.nanmin(g)


def find_constant_phi_separator(cv1: np.ndarray, cv2: np.ndarray, F: np.ndarray) -> tuple[float, float]:
    """Return a symmetric |CNNC| separator from the marginal F(CNNC).

    Cis is centered around CNNC=0 deg and trans at the periodic edges. The split
    is therefore |phi| <= phi_sep for cis and the complement for trans.
    """
    g_phi = marginal_phi_free_energy(cv1, cv2, F)
    # The cis basin is centered at CNNC ~= 0 deg and trans at the periodic
    # edges. The chemically meaningful dividing plane should sit near the
    # rotational barrier, not inside the cis well; use a broad but guarded
    # search interval around the expected 90 deg separation.
    positive = np.where((cv1 >= 60.0) & (cv1 <= 140.0))[0]
    if positive.size == 0:
        raise ValueError("No positive CNNC grid points available for separator search.")
    idx = int(positive[np.nanargmax(g_phi[positive])])
    return float(abs(cv1[idx])), float(g_phi[idx])


def marginal_values_at_separators(cv1: np.ndarray, cv2: np.ndarray, F: np.ndarray, phi_sep: float) -> tuple[float, float]:
    g_phi = marginal_phi_free_energy(cv1, cv2, F)
    idx_pos = int(np.argmin(np.abs(cv1 - phi_sep)))
    idx_neg = int(np.argmin(np.abs(cv1 + phi_sep)))
    return float(g_phi[idx_pos]), float(g_phi[idx_neg])


def consensus_separators() -> dict[str, float]:
    separators: dict[str, float] = {}
    for system in SYSTEMS:
        marginal_curves = []
        cv1_ref = None
        for model in MODELS:
            cv1, cv2, F, _ = load_surface(system, model)
            cv1_ref = cv1
            marginal_curves.append(marginal_phi_free_energy(cv1, cv2, F))
        if cv1_ref is None:
            raise RuntimeError(f"No surfaces available for {system}")
        average_marginal = np.mean(np.vstack(marginal_curves), axis=0)
        search = np.where((cv1_ref >= 60.0) & (cv1_ref <= 140.0))[0]
        separators[system] = float(abs(cv1_ref[int(search[np.nanargmax(average_marginal[search])])]))
    return separators


def basin_stats(
    cv1: np.ndarray,
    cv2: np.ndarray,
    F: np.ndarray,
    phi_sep: float,
    separator_pos_f_phi: float,
    separator_neg_f_phi: float,
) -> dict[str, float]:
    phi_abs = np.abs(cv1)[:, None]
    cis_mask = np.broadcast_to(phi_abs <= phi_sep, F.shape)
    trans_mask = ~cis_mask
    beta = 1.0 / (KB_EV_PER_K * TEMPERATURE_K)
    z_cis = boltzmann_partition(F, cv1, cv2, cis_mask)
    z_trans = boltzmann_partition(F, cv1, cv2, trans_mask)
    g_cis = -np.log(z_cis) / beta
    g_trans = -np.log(z_trans) / beta
    offset = min(g_cis, g_trans)
    g_cis -= offset
    g_trans -= offset
    separator_max_f_phi = max(separator_pos_f_phi, separator_neg_f_phi)
    return {
        "phi_separator_deg": phi_sep,
        "separator_pos_phi_deg": phi_sep,
        "separator_neg_phi_deg": -phi_sep,
        "separator_pos_marginal_F_eV": separator_pos_f_phi,
        "separator_neg_marginal_F_eV": separator_neg_f_phi,
        "separator_max_marginal_F_eV": separator_max_f_phi,
        "cis_basin_G_eV": float(g_cis),
        "trans_basin_G_eV": float(g_trans),
        "deltaG_cis_minus_trans_eV": float(g_cis - g_trans),
        "barrier_pos_from_cis_basin_eV": float(separator_pos_f_phi - g_cis),
        "barrier_neg_from_cis_basin_eV": float(separator_neg_f_phi - g_cis),
        "barrier_pos_from_trans_basin_eV": float(separator_pos_f_phi - g_trans),
        "barrier_neg_from_trans_basin_eV": float(separator_neg_f_phi - g_trans),
        "barrier_max_from_cis_basin_eV": float(separator_max_f_phi - g_cis),
        "barrier_max_from_trans_basin_eV": float(separator_max_f_phi - g_trans),
    }


def add_units(row: dict[str, float]) -> dict[str, float]:
    out = dict(row)
    for key, value in row.items():
        if key.endswith("_eV"):
            out[key.replace("_eV", "_kJmol")] = value * EV_TO_KJMOL
            out[key.replace("_eV", "_kcalmol")] = value * EV_TO_KCALMOL
    return out


def compute_all() -> tuple[list[dict[str, str | float]], dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]]:
    rows: list[dict[str, str | float]] = []
    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = {}
    separators = consensus_separators()
    for system in SYSTEMS:
        for model in MODELS:
            cv1, cv2, F, actual_ns = load_surface(system, model)
            phi_sep = separators[system]
            marginal_pos, marginal_neg = marginal_values_at_separators(cv1, cv2, F, phi_sep)
            stats = add_units(basin_stats(cv1, cv2, F, phi_sep, marginal_pos, marginal_neg))
            rows.append(
                {
                    "system": system,
                    "model": model,
                    "label": LABELS[model],
                    "temperature_K": TEMPERATURE_K,
                    "actual_time_ns": actual_ns,
                    "separator_strategy": "consensus_average_marginal_pmf_by_starting_ensemble",
                    **stats,
                }
            )
            surfaces[(system, model)] = (cv1, cv2, F, actual_ns, phi_sep)
    return rows, surfaces


def write_csv(rows: list[dict[str, str | float]]) -> Path:
    out = OUT_DIR / "azobenzene_2d_boltzmann_basin_free_energies.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


def write_summary_csv(rows: list[dict[str, str | float]]) -> Path:
    out = OUT_DIR / "azobenzene_2d_boltzmann_barrier_summary_all_channels.csv"
    summary_rows: list[dict[str, str | float | int]] = []
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        if not subset:
            continue
        c_to_t_values = np.asarray(
            [
                float(row[column])
                for row in subset
                for column in ("barrier_pos_from_cis_basin_eV", "barrier_neg_from_cis_basin_eV")
            ],
            dtype=float,
        )
        t_to_c_values = np.asarray(
            [
                float(row[column])
                for row in subset
                for column in ("barrier_pos_from_trans_basin_eV", "barrier_neg_from_trans_basin_eV")
            ],
            dtype=float,
        )
        delta_g_values = np.asarray([float(row["deltaG_cis_minus_trans_eV"]) for row in subset], dtype=float)
        summary_rows.append(
            {
                "model": model,
                "label": LABELS[model],
                "n_starting_ensembles": len(subset),
                "n_channels_per_direction": len(c_to_t_values),
                "deltaG_cis_minus_trans_eV_mean": float(np.mean(delta_g_values)),
                "deltaG_cis_minus_trans_eV_std": float(np.std(delta_g_values, ddof=1)) if len(delta_g_values) > 1 else 0.0,
                "barrier_cis_to_trans_eV_mean": float(np.mean(c_to_t_values)),
                "barrier_cis_to_trans_eV_std": float(np.std(c_to_t_values, ddof=1)) if len(c_to_t_values) > 1 else 0.0,
                "barrier_trans_to_cis_eV_mean": float(np.mean(t_to_c_values)),
                "barrier_trans_to_cis_eV_std": float(np.std(t_to_c_values, ddof=1)) if len(t_to_c_values) > 1 else 0.0,
                "deltaG_cis_minus_trans_kJmol_mean": float(np.mean(delta_g_values) * EV_TO_KJMOL),
                "deltaG_cis_minus_trans_kJmol_std": float(
                    (np.std(delta_g_values, ddof=1) if len(delta_g_values) > 1 else 0.0) * EV_TO_KJMOL
                ),
                "barrier_cis_to_trans_kJmol_mean": float(np.mean(c_to_t_values) * EV_TO_KJMOL),
                "barrier_cis_to_trans_kJmol_std": float(
                    (np.std(c_to_t_values, ddof=1) if len(c_to_t_values) > 1 else 0.0) * EV_TO_KJMOL
                ),
                "barrier_trans_to_cis_kJmol_mean": float(np.mean(t_to_c_values) * EV_TO_KJMOL),
                "barrier_trans_to_cis_kJmol_std": float(
                    (np.std(t_to_c_values, ddof=1) if len(t_to_c_values) > 1 else 0.0) * EV_TO_KJMOL
                ),
            }
        )
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return out


def plot_system(system: str, surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]) -> None:
    fig = plt.figure(figsize=(15.0, 8.2), facecolor="white")
    panel_width = 0.185
    panel_height = 0.385
    top_y = 0.555
    bottom_y = 0.075
    x_positions = (0.035, 0.255, 0.475, 0.695)
    positions = tuple((x, top_y, panel_width, panel_height) for x in x_positions) + tuple(
        (x, bottom_y, panel_width, panel_height) for x in x_positions
    )
    vmax = float(np.ceil(max(np.nanmax(surfaces[(system, model)][2]) for model in MODELS) * 10.0) / 10.0)
    levels = np.linspace(0.0, vmax, PLOT_LEVELS)
    contour = None
    for panel_index, (model, position) in enumerate(zip(MODELS, positions, strict=True)):
        ax = fig.add_axes(position)
        cv1, cv2, F, actual_ns, phi_sep = surfaces[(system, model)]
        contour = ax.contourf(cv1, cv2, F.T, levels=levels, cmap="RdBu_r")
        for x in (-phi_sep, phi_sep):
            ax.axvline(x, color="white", lw=2.8, ls="--", zorder=5)
            ax.axvline(x, color="black", lw=1.4, ls="--", zorder=6)
        ax.set_title(f"{LABELS[model]}\n{actual_ns:.2f} ns", pad=6)
        ax.set_xlim(-180, 180)
        ax.set_ylim(60, 180)
        ax.set_xticks((-180, -90, 0, 90, 180))
        ax.set_yticks((60, 90, 120, 150, 180))
        ax.set_xlabel(r"CNNC dihedral, $\phi$ (deg)")
        ax.set_ylabel(r"CNN angle, $\theta$ (deg)" if panel_index in (0, 4) else "")
        ax.tick_params(direction="in", top=True, right=True, length=4.5, width=1.0)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5, width=0.8)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
    colorbar_ax = fig.add_axes((0.925, bottom_y, 0.018, top_y + panel_height - bottom_y))
    cbar = fig.colorbar(contour, cax=colorbar_ax)
    cbar.set_label("Relative free energy (eV)")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.outline.set_linewidth(1.0)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"azobenzene_{system}_2d_boltzmann_separator.{ext}",
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
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, surfaces = compute_all()
    out = write_csv(rows)
    summary = write_summary_csv(rows)
    for system in SYSTEMS:
        plot_system(system, surfaces)
    print(out)
    print(summary)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
