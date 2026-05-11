from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
OUT_CSV = ROOT / "analysis" / "barrier_overview_malonaldehyd_models.csv"


@dataclass
class Entry:
    model_key: str
    family: str
    source: str
    system: str
    csv_path: Path


def system_entries() -> list[Entry]:
    entries: list[Entry] = []
    mace_map = {
        "off_large": ("MACE", "off_large"),
        "off_medium": ("MACE", "off_medium"),
        "off_small": ("MACE", "off_small"),
        "off24_medium": ("MACE", "off24_medium"),
        "omol_extra_large": ("MACE", "omol_extra_large"),
        "mh1_mh-1": ("MACE", "mh1_mh-1"),
        "polar_l": ("MACE", "polar_l"),
        "polar_m": ("MACE", "polar_m"),
        "polar_s": ("MACE", "polar_s"),
    }
    for system in ("malonaldehyd", "f-malonaldehyd"):
        analysis_dir = ROOT / system / "analysis"
        for stem, (family, model_key) in mace_map.items():
            entries.append(
                Entry(
                    model_key=model_key,
                    family=family,
                    source="raccoon",
                    system=system,
                    csv_path=analysis_dir / f"umbrella_integration_{stem}.csv",
                )
            )

    for system in ("malonaldehyd", "f-malonaldehyd"):
        entries.append(
            Entry(
                model_key="upet_pet-oam-xl",
                family="UPET",
                source="viper",
                system=system,
                csv_path=ROOT / "viper_analysis" / "upet" / system / "umbrella_integration_viper_upet.csv",
            )
        )
        entries.append(
            Entry(
                model_key="upet_pet-spice-l",
                family="UPET",
                source="viper",
                system=system,
                csv_path=ROOT
                / "viper_analysis"
                / "upet"
                / system
                / "umbrella_integration_viper_upet_pet_spice.csv",
            )
        )
        entries.append(
            Entry(
                model_key="sol3r",
                family="SO3LR",
                source="viper",
                system=system,
                csv_path=ROOT / "viper_analysis" / "sol3r" / system / "umbrella_integration_viper_sol3r.csv",
            )
        )
    return entries


def central_barriers(csv_path: Path) -> dict[str, float]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    x = np.atleast_1d(data["mean_cv"]).astype(float)
    y = np.atleast_1d(data["free_energy"]).astype(float)

    central_mask = np.abs(x) <= 0.2
    if np.any(central_mask):
        central_indices = np.where(central_mask)[0]
        barrier_idx = int(central_indices[np.argmax(y[central_indices])])
    else:
        barrier_idx = int(np.argmax(y))

    left_idx = int(np.argmin(y[: barrier_idx + 1]))
    right_idx = barrier_idx + int(np.argmin(y[barrier_idx:]))

    return {
        "left_min_cv": float(x[left_idx]),
        "left_min_fe": float(y[left_idx]),
        "barrier_cv": float(x[barrier_idx]),
        "barrier_fe": float(y[barrier_idx]),
        "right_min_cv": float(x[right_idx]),
        "right_min_fe": float(y[right_idx]),
        "left_barrier_height": float(y[barrier_idx] - y[left_idx]),
        "right_barrier_height": float(y[barrier_idx] - y[right_idx]),
    }


def main() -> None:
    OUT_CSV.parent.mkdir(exist_ok=True)
    by_model: dict[tuple[str, str, str], dict[str, object]] = {}

    for entry in system_entries():
        if not entry.csv_path.exists():
            continue
        key = (entry.family, entry.model_key, entry.source)
        row = by_model.setdefault(
            key,
            {
                "family": entry.family,
                "model": entry.model_key,
                "source": entry.source,
            },
        )
        stats = central_barriers(entry.csv_path)
        prefix = "malon" if entry.system == "malonaldehyd" else "f_malon"
        for name in (
            "left_barrier_height",
            "right_barrier_height",
            "left_min_cv",
            "barrier_cv",
            "right_min_cv",
        ):
            row[f"{prefix}_{name}"] = stats[name]
        row[f"{prefix}_csv"] = str(entry.csv_path)

    fieldnames = [
        "family",
        "model",
        "source",
        "malon_left_barrier_height",
        "malon_right_barrier_height",
        "malon_left_min_cv",
        "malon_barrier_cv",
        "malon_right_min_cv",
        "f_malon_left_barrier_height",
        "f_malon_right_barrier_height",
        "f_malon_left_min_cv",
        "f_malon_barrier_cv",
        "f_malon_right_min_cv",
        "malon_csv",
        "f_malon_csv",
    ]

    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(by_model):
            writer.writerow(by_model[key])

    print(OUT_CSV)


if __name__ == "__main__":
    main()
