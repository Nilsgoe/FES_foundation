from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
AZOB_ANALYSIS = ROOT / "azobenzene" / "analysis"
VIPER_ANALYSIS = ROOT / "viper_analysis"
OUT_DIR = AZOB_ANALYSIS / "model_comparison"
OUT_CSV = OUT_DIR / "azobenzene_1d_barriers.csv"


COLORS = {
    "off": "#1b9e77",
    "omol": "#d95f02",
    "mh1": "#7570b3",
    "polar": "#e7298a",
    "upet": "#1f78b4",
    "sol3r": "#e6ab02",
}

LABELS = {
    "off": "MACE-OFF",
    "omol": "MACE-OMOL",
    "mh1": "MACE-MH1",
    "polar": "MACE-Polar",
    "upet": "UPET",
    "sol3r": "SO3LR",
}


@dataclass
class FESRecord:
    system: str
    model: str
    source: str
    path: Path


def load_records() -> list[FESRecord]:
    records: list[FESRecord] = []
    for system in ("cis", "trans"):
        for model in ("off", "omol", "mh1", "polar"):
            records.append(
                FESRecord(
                    system=system,
                    model=model,
                    source="raccoon",
                    path=AZOB_ANALYSIS / f"metad_azob_{system}_{model}_1d_raccoon_{model}_job"
                    f"{'33975' if model == 'off' else '33976' if model == 'omol' else '33030' if model == 'mh1' else '33031'}"
                    f"_task{'0' if system == 'cis' else '1'}_gpu0_reconstructed_fes.csv",
                )
            )
        for model in ("upet", "sol3r"):
            records.append(
                FESRecord(
                    system=system,
                    model=model,
                    source="viper",
                    path=VIPER_ANALYSIS / model / "azobenzene" / f"{system}_1d" / f"{model}_azob_{system}_1d_reconstructed_fes.csv",
                )
            )
    return records


def load_fes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data["cv"]).astype(float), np.atleast_1d(data["free_energy"]).astype(float)


def cis_barriers(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    cis_mask = np.abs(x) <= 60.0
    left_barrier_mask = (x >= -150.0) & (x <= -30.0)
    right_barrier_mask = (x >= 30.0) & (x <= 150.0)
    trans_mask = np.abs(x) >= 150.0

    cis_indices = np.where(cis_mask)[0]
    left_indices = np.where(left_barrier_mask)[0]
    right_indices = np.where(right_barrier_mask)[0]
    trans_indices = np.where(trans_mask)[0]

    cis_idx = int(cis_indices[np.argmin(y[cis_indices])])
    left_idx = int(left_indices[np.argmax(y[left_indices])])
    right_idx = int(right_indices[np.argmax(y[right_indices])])
    trans_idx = int(trans_indices[np.argmin(y[trans_indices])])

    cis_left = None
    cis_right = None
    if y[cis_idx] < y[left_idx] and y[cis_idx] < y[right_idx]:
        cis_left = float(y[left_idx] - y[cis_idx])
        cis_right = float(y[right_idx] - y[cis_idx])

    return {
        "cis_min_cv": float(x[cis_idx]),
        "cis_min_fe": float(y[cis_idx]),
        "left_barrier_cv": float(x[left_idx]),
        "left_barrier_fe": float(y[left_idx]),
        "right_barrier_cv": float(x[right_idx]),
        "right_barrier_fe": float(y[right_idx]),
        "left_barrier_height_from_cis": cis_left,
        "right_barrier_height_from_cis": cis_right,
        "left_barrier_height_from_trans": float(y[left_idx] - y[trans_idx]),
        "right_barrier_height_from_trans": float(y[right_idx] - y[trans_idx]),
        "trans_min_cv": float(x[trans_idx]),
        "trans_min_fe": float(y[trans_idx]),
    }


def make_plot(system: str, records: list[FESRecord]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for record in records:
        x, y = load_fes(record.path)
        label = LABELS[record.model]
        if record.source == "viper":
            label = f"{label} ({record.source})"
        ax.plot(x, y, label=label, color=COLORS[record.model], linewidth=2.4)

    ax.set_xlim(-180, 180)
    ax.set_xlabel("CNNC dihedral (deg)")
    ax.set_ylabel("Relative free energy (arb. units)")
    ax.set_title(f"Azobenzene {system} 1D MetaD comparison")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    fig.tight_layout()

    outputs: list[Path] = []
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"azobenzene_{system}_1d_model_comparison.{ext}"
        fig.savefig(out, dpi=300)
        outputs.append(out)
    plt.close(fig)
    return outputs


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.3,
            "lines.linewidth": 2.4,
            "savefig.bbox": "tight",
        }
    )
    OUT_DIR.mkdir(exist_ok=True)
    records = [r for r in load_records() if r.path.exists()]

    for system in ("cis", "trans"):
        make_plot(system, [r for r in records if r.system == system])

    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "system",
                "model",
                "source",
                "left_barrier_height_from_trans",
                "right_barrier_height_from_trans",
                "left_barrier_height_from_cis",
                "right_barrier_height_from_cis",
                "cis_min_cv",
                "left_barrier_cv",
                "right_barrier_cv",
                "trans_min_cv",
                "csv_path",
            ]
        )
        for record in sorted(records, key=lambda r: (r.system, r.model)):
            x, y = load_fes(record.path)
            stats = cis_barriers(x, y)
            writer.writerow(
                [
                    record.system,
                    record.model,
                    record.source,
                    stats["left_barrier_height_from_trans"],
                    stats["right_barrier_height_from_trans"],
                    stats["left_barrier_height_from_cis"],
                    stats["right_barrier_height_from_cis"],
                    stats["cis_min_cv"],
                    stats["left_barrier_cv"],
                    stats["right_barrier_cv"],
                    stats["trans_min_cv"],
                    str(record.path),
                ]
            )

    print(OUT_CSV)


if __name__ == "__main__":
    main()
