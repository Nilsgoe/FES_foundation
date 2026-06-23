from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import extend_malon_analysis as ema


def load_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    data = np.atleast_1d(data)
    names = data.dtype.names or ()
    if "mean_cv" in names:
        x = np.asarray(data["mean_cv"], dtype=float)
    elif "cv" in names:
        x = np.asarray(data["cv"], dtype=float)
    else:
        raise ValueError(f"No CV column found in {csv_path}")

    if "free_energy_ui" in names:
        y = np.asarray(data["free_energy_ui"], dtype=float)
    elif "free_energy" in names:
        y = np.asarray(data["free_energy"], dtype=float)
    else:
        raise ValueError(f"No umbrella free-energy column found in {csv_path}")

    if "mean_force" in names:
        force = np.asarray(data["mean_force"], dtype=float)
    else:
        force = np.full_like(x, np.nan, dtype=float)
    return x, y - np.nanmin(y), force


def plot_umbrella(csv_path: Path, model_tag: str, output_dir: Path) -> list[Path]:
    x, free_energy, force = load_curve(csv_path)
    color = ema.MODEL_COLORS.get(model_tag, "#1f78b4")
    label = ema.MODEL_LABELS.get(model_tag, model_tag.replace("_", " "))

    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    ax1.plot(x, free_energy, marker="o", color=color, label=f"{label} UI PMF")
    ax1.set_xlabel("CV (A)")
    ax1.set_ylabel("Relative free energy (eV)")
    ax1.set_title(f"{output_dir.parent.name}: {label}")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, force, color="#444444", linestyle="--", linewidth=1.8, label="Mean force")
    ax2.set_ylabel("Mean force (eV/A)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=False)
    fig.tight_layout()

    outputs: list[Path] = []
    for ext in ("png", "pdf"):
        out = output_dir / f"umbrella_integration_{model_tag}.{ext}"
        fig.savefig(out, dpi=300)
        outputs.append(out)
    plt.close(fig)
    return outputs


def iter_targets() -> list[tuple[Path, str, Path]]:
    targets: list[tuple[Path, str, Path]] = []
    for system_name in ema.SYSTEMS:
        analysis_dir = ema.ROOT / system_name / "analysis"
        for csv_path in sorted(analysis_dir.glob("umbrella_integration_*.csv")):
            if csv_path.stem.endswith("_analysis"):
                continue
            model_tag = csv_path.stem.replace("umbrella_integration_", "")
            targets.append((csv_path, model_tag, analysis_dir))

        for model_tag, csv_path, _raw_dir in ema.VIPER_ANALYSIS_INPUTS[system_name]:
            if csv_path.exists():
                targets.append((csv_path, model_tag, csv_path.parent))
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate normal malonaldehyde umbrella-integration plots.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    ema.plot_style()
    for csv_path, model_tag, output_dir in iter_targets():
        generated = plot_umbrella(csv_path, model_tag, output_dir)
        for path in generated:
            print(path)


if __name__ == "__main__":
    main()
