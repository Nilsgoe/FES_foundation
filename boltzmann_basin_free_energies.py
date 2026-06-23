from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
OUT_CSV = ROOT / "analysis" / "boltzmann_basin_free_energies_malonaldehyd.csv"
KB_EV_PER_K = 8.617333262145e-5
EV_TO_KJMOL = 96.48533212
EV_TO_KCALMOL = 23.06054887


@dataclass(frozen=True)
class Entry:
    family: str
    model: str
    source: str
    system: str
    temperature_k: float
    path: Path


def entries() -> list[Entry]:
    rows: list[Entry] = []
    mace = {
        "off24_medium": ("MACE", "MACE-OFF24 M"),
        "mh1_mh-1": ("MACE", "MACE-MH1"),
        "polar_m": ("MACE", "MACE-Polar M"),
    }
    for system in ("malonaldehyd", "f-malonaldehyd"):
        for stem, (family, label) in mace.items():
            rows.append(
                Entry(
                    family=family,
                    model=label,
                    source="raccoon",
                    system=system,
                    temperature_k=293.0,
                    path=ROOT / system / "analysis" / f"umbrella_integration_{stem}.csv",
                )
            )
        rows.extend(
            [
                Entry(
                    "SO3LR",
                    "SO3LR",
                    "viper",
                    system,
                    293.0,
                    ROOT / "viper_analysis" / "sol3r" / system / "umbrella_integration_viper_sol3r.csv",
                ),
                Entry(
                    "UPET",
                    "PET-SPICE",
                    "viper",
                    system,
                    293.0,
                    ROOT / "viper_analysis" / "upet" / system / "umbrella_integration_viper_upet_pet_spice.csv",
                ),
                Entry(
                    "UPET",
                    "PET-SPICE_rot",
                    "viper",
                    system,
                    293.0,
                    ROOT
                    / "viper_analysis"
                    / "upet"
                    / f"{system}_pet_spice_rotavg3_43w"
                    / "umbrella_integration_viper_upet_pet_spice_rot.csv",
                ),
            ]
        )
    rows.extend(
        [
            Entry(
                "DFT",
                "wB97M-V/def2-TZVPD",
                "viper_orca",
                "malonaldehyd",
                300.0,
                ROOT / "analysis/orca_dft/malonaldehyd_8ps/umbrella_integration_orca_dft_8ps.csv",
            ),
            Entry(
                "DFT",
                "wB97M-V/def2-TZVPD",
                "viper_orca",
                "f-malonaldehyd",
                300.0,
                ROOT
                / "analysis/orca_dft/f-malonaldehyd_4p5ps_mlip_cv"
                / "umbrella_integration_orca_dft_4p5ps_mlip_cv.csv",
            ),
        ]
    )
    return rows


def read_pmf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names or ()
    x = np.atleast_1d(data["mean_cv"]).astype(float)
    for column in ("free_energy_diff_gp", "free_energy", "free_energy_ui"):
        if column in names:
            y = np.atleast_1d(data[column]).astype(float)
            break
    else:
        raise ValueError(f"No supported free-energy column in {path}")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    y = y - np.nanmin(y)
    return x, y


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    if x.size < 2:
        return np.ones_like(x)
    widths = np.empty_like(x)
    widths[1:-1] = 0.5 * (x[2:] - x[:-2])
    widths[0] = 0.5 * (x[1] - x[0])
    widths[-1] = 0.5 * (x[-1] - x[-2])
    return np.clip(widths, np.finfo(float).tiny, None)


def central_barrier_index(x: np.ndarray, y: np.ndarray) -> int:
    central = np.where(np.abs(x) <= 0.2)[0]
    if central.size:
        return int(central[np.argmax(y[central])])
    return int(np.argmax(y))


def basin_free_energy(x: np.ndarray, y: np.ndarray, mask: np.ndarray, temperature_k: float) -> float:
    beta = 1.0 / (KB_EV_PER_K * temperature_k)
    weights = trapezoid_weights(x)
    shifted = y - np.nanmin(y)
    z = np.sum(weights[mask] * np.exp(-beta * shifted[mask]))
    return float(-np.log(z) / beta)


def analyze(path: Path, temperature_k: float) -> dict[str, float]:
    x, y = read_pmf(path)
    idx_barrier = central_barrier_index(x, y)
    left_mask = np.arange(x.size) <= idx_barrier
    right_mask = np.arange(x.size) >= idx_barrier
    g_left = basin_free_energy(x, y, left_mask, temperature_k)
    g_right = basin_free_energy(x, y, right_mask, temperature_k)
    offset = min(g_left, g_right)
    g_left -= offset
    g_right -= offset
    f_barrier = float(y[idx_barrier])
    left_point_idx = int(np.argmin(y[: idx_barrier + 1]))
    right_point_idx = idx_barrier + int(np.argmin(y[idx_barrier:]))
    return {
        "separator_cv_A": float(x[idx_barrier]),
        "separator_F_eV": f_barrier,
        "left_min_cv_A": float(x[left_point_idx]),
        "right_min_cv_A": float(x[right_point_idx]),
        "left_point_min_F_eV": float(y[left_point_idx]),
        "right_point_min_F_eV": float(y[right_point_idx]),
        "left_basin_G_eV": g_left,
        "right_basin_G_eV": g_right,
        "deltaG_left_minus_right_eV": g_left - g_right,
        "barrier_from_left_basin_eV": f_barrier - g_left,
        "barrier_from_right_basin_eV": f_barrier - g_right,
    }


def with_units(row: dict[str, float]) -> dict[str, float]:
    out = dict(row)
    for key, value in row.items():
        if key.endswith("_eV"):
            out[key.replace("_eV", "_kJmol")] = value * EV_TO_KJMOL
            out[key.replace("_eV", "_kcalmol")] = value * EV_TO_KCALMOL
    return out


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    for entry in entries():
        if not entry.path.exists():
            continue
        stats = with_units(analyze(entry.path, entry.temperature_k))
        rows.append(
            {
                "system": entry.system,
                "family": entry.family,
                "model": entry.model,
                "source": entry.source,
                "temperature_K": entry.temperature_k,
                "csv": str(entry.path),
                **stats,
            }
        )
    if not rows:
        raise SystemExit("No inputs found.")
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
