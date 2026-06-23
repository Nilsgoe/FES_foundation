from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azobenzene.scripts.barrier_extraction.fes_io import parse_bias_log


OUT_DIR = ROOT / "azobenzene" / "analysis" / "model_comparison" / "2ns_8model_paper"
MAX_TIME_FS = 2_000_000.0
EV_TO_KJMOL = 96.485

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
COLORS = {
    "off": "#1f77b4",
    "mh1": "#ff7f0e",
    "polar": "#2ca02c",
    "pet_spice": "#d62728",
    "so3lr": "#9467bd",
    "scratch": "#8c564b",
    "ft_so3lr": "#17becf",
    "ft_mh1": "#bcbd22",
}


@dataclass(frozen=True)
class Record:
    system: str
    model: str
    path: Path


def _pet_spice_candidates(system: str) -> tuple[Path, ...]:
    return (
        Path("/work/gpuviper_ptmp/Enhanced_sampling/upet/azobenzene")
        / f"pet_spice_{system}_2d"
        / "outputs"
        / f"pet_spice_azob_{system}_2d.bias",
        ROOT
        / "viper_analysis"
        / "upet"
        / "azobenzene"
        / f"pet_spice_{system}_2d"
        / f"pet_spice_azob_{system}_2d.bias",
    )


def _choose_longest_existing(paths: tuple[Path, ...]) -> Path:
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError("No existing candidate files:\n" + "\n".join(map(str, paths)))
    if len(existing) == 1:
        return existing[0]
    parsed = [(float(parse_bias_log(path).time_fs[-1]), path) for path in existing]
    return max(parsed, key=lambda item: item[0])[1]


def load_records() -> list[Record]:
    jobs = {"off": ("off", "33975"), "mh1": ("mh1", "33030"), "polar": ("polar", "33031")}
    records: list[Record] = []
    for system in SYSTEMS:
        task = "0" if system == "cis" else "1"
        for model, (filename_model, job) in jobs.items():
            records.append(
                Record(
                    system,
                    model,
                    ROOT
                    / "azobenzene"
                    / "outputs"
                    / f"metad_azob_{system}_{filename_model}_2d_raccoon_{filename_model}_job{job}_task{task}_gpu1.txt",
                )
            )
        records.extend(
            [
                Record(system, "pet_spice", _choose_longest_existing(_pet_spice_candidates(system))),
                Record(
                    system,
                    "so3lr",
                    ROOT
                    / "viper_analysis"
                    / "sol3r"
                    / "azobenzene"
                    / f"{system}_2d"
                    / f"sol3r_azob_{system}_2d.bias",
                ),
                Record(
                    system,
                    "scratch",
                    ROOT / "raccoon_mace_scratch_azob" / "outputs" / f"mace_scratch_azob_{system}_2d.bias",
                ),
                Record(
                    system,
                    "ft_so3lr",
                    ROOT / "viper_ft_so3lr_azob" / "outputs" / f"ft_so3lr_azob_{system}_2d.bias",
                ),
                Record(
                    system,
                    "ft_mh1",
                    ROOT
                    / "raccoon_mace_ft_mh1_azob"
                    / "outputs"
                    / f"mace_ft_mh1_azob_{system}_2d.bias",
                ),
            ]
        )
    return records


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()

    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.6), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.ravel()
    summary_rows = []
    for ax, model in zip(axes_flat, MODELS, strict=True):
        for system, linestyle in (("cis", "-"), ("trans", "--")):
            record = next(item for item in records if item.system == system and item.model == model)
            run = parse_bias_log(record.path)
            n_hills = min(len(run.time_fs), int(np.searchsorted(run.time_fs, MAX_TIME_FS, side="right")))
            time_ns = run.time_fs[:n_hills] / 1e6
            height_kjmol = run.height_eV[:n_hills] * EV_TO_KJMOL
            ax.plot(time_ns, height_kjmol, color=COLORS[model], lw=1.8, alpha=0.95, ls=linestyle, label=f"{system} start")
            summary_rows.append(
                {
                    "system": system,
                    "model": model,
                    "label": LABELS[model],
                    "n_hills": n_hills,
                    "last_time_ns": float(time_ns[-1]),
                    "initial_height_kJmol": float(height_kjmol[0]),
                    "final_height_kJmol": float(height_kjmol[-1]),
                    "source": str(record.path),
                }
            )
        ax.set_title(LABELS[model])
        ax.set_xlim(0.0, 2.0)
        ax.set_yscale("log")
        ax.tick_params(direction="in", top=True, right=True, length=4.5, width=1.0)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5, width=0.8)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
    for ax in axes[-1, :]:
        ax.set_xlabel("Simulation time (ns)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Deposited hill height (kJ mol$^{-1}$)")
    axes[0, -1].legend(loc="upper right", frameon=False)

    for extension in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"azobenzene_2d_8model_bias_height_vs_time.{extension}",
            dpi=600,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)

    import csv

    with (OUT_DIR / "azobenzene_2d_8model_bias_height_vs_time_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {OUT_DIR / 'azobenzene_2d_8model_bias_height_vs_time.png'}")
    print(f"Wrote {OUT_DIR / 'azobenzene_2d_8model_bias_height_vs_time.pdf'}")
    print(f"Wrote {OUT_DIR / 'azobenzene_2d_8model_bias_height_vs_time_summary.csv'}")


if __name__ == "__main__":
    main()
