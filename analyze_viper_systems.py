from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORK_ROOT = Path("/work/gpuviper_ptmp/Enhanced_sampling")
REPO_ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
OUTPUT_ROOT = REPO_ROOT / "viper_analysis"

SYSTEM_ROOTS = {
    "upet": WORK_ROOT / "upet",
    "sol3r": WORK_ROOT / "sol3r",
}

AZOB_CASES = ("cis_1d", "cis_2d", "trans_1d", "trans_2d")
UMD_CASES = ("malonaldehyd", "f-malonaldehyd")

UMD_MEAN_RE = re.compile(r"mean_cv_energy_(.+)_shift_(-?\d+)\.csv$")
HEADER_RE = re.compile(r"([A-Za-z_]+)=([A-Za-z0-9.+-]+)")

MODEL_LABELS = {
    "off_large": "MACE-OFF large",
    "off_medium": "MACE-OFF medium",
    "off_small": "MACE-OFF small",
    "off24_medium": "MACE-OFF24 medium",
    "omol_extra_large": "MACE-OMOL XL",
    "mh1_mh-1": "MACE-MH1",
    "polar_l": "MACE-Polar L",
    "polar_m": "MACE-Polar M",
    "polar_s": "MACE-Polar S",
    "viper_upet": "UPET PET-OAM-XL",
    "viper_upet_pet_spice": "UPET PET-SPICE-L",
    "viper_sol3r": "SO3LR",
}

MODEL_COLORS = {
    "off_large": "#1b9e77",
    "off_medium": "#66a61e",
    "off_small": "#a6d854",
    "off24_medium": "#2ca25f",
    "omol_extra_large": "#d95f02",
    "mh1_mh-1": "#7570b3",
    "polar_l": "#e7298a",
    "polar_m": "#e6ab02",
    "polar_s": "#1f78b4",
    "viper_upet": "#fb9a99",
    "viper_upet_pet_spice": "#e31a1c",
    "viper_sol3r": "#6a3d9a",
}


@dataclass
class UmbrellaWindow:
    center: float
    mean_cv: float
    mean_force: float
    mean_energy: float
    initial_cv: float
    shift: int
    source: Path


def parse_mean_csv(path: Path) -> tuple[str, UmbrellaWindow] | None:
    match = UMD_MEAN_RE.match(path.name)
    if not match:
        return None
    model_tag = match.group(1)
    shift = int(match.group(2))
    with path.open() as handle:
        row = next(csv.reader(handle))
    values = [float(x) for x in row]
    if len(values) < 5:
        return None
    return model_tag, UmbrellaWindow(
        center=values[0],
        mean_cv=values[1],
        mean_force=values[2],
        mean_energy=values[3],
        initial_cv=values[4],
        shift=shift,
        source=path,
    )


def load_umbrella_windows(paths: list[Path]) -> list[UmbrellaWindow]:
    windows: list[UmbrellaWindow] = []
    for path in paths:
        with path.open() as handle:
            row = next(csv.reader(handle))
        values = [float(x) for x in row]
        if len(values) < 5:
            continue
        shift_match = re.search(r"_shift_(-?\d+)(?:_|\.csv$)", path.name)
        shift = int(shift_match.group(1)) if shift_match else 0
        windows.append(
            UmbrellaWindow(
                center=values[0],
                mean_cv=values[1],
                mean_force=values[2],
                mean_energy=values[3],
                initial_cv=values[4],
                shift=shift,
                source=path,
            )
        )
    return sorted(windows, key=lambda w: w.mean_cv)


def write_umbrella_outputs(model_tag: str, windows: list[UmbrellaWindow], output_dir: Path) -> list[str]:
    if not windows:
        return []

    x = np.array([w.mean_cv for w in windows], dtype=float)
    force = np.array([w.mean_force for w in windows], dtype=float)
    free_energy = cumulative_trapezoid(x, force)
    free_energy -= np.min(free_energy)

    csv_path = output_dir / f"umbrella_integration_{model_tag}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mean_cv", "free_energy", "mean_force", "window_center", "shift", "source_file"])
        for window, xi, fi, dfi in zip(windows, x, free_energy, force):
            writer.writerow([xi, fi, dfi, window.center, window.shift, window.source.name])

    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    ax1.plot(x, free_energy, marker="o", color="#1f78b4", label="Umbrella integration FES")
    ax1.set_xlabel("CV (A)")
    ax1.set_ylabel("Relative free energy (arb. units)")
    ax1.set_title(f"{output_dir.parent.name}/{output_dir.name}: {model_tag}")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, force, color="#d95f02", linestyle="--", linewidth=2.0, label="Mean force")
    ax2.set_ylabel("Mean force")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=False)
    fig.tight_layout()

    png_path = output_dir / f"umbrella_integration_{model_tag}.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return [str(csv_path), str(png_path)]


def cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    if len(x) < 2:
        return out
    out[1:] = np.cumsum(np.diff(x) * 0.5 * (y[1:] + y[:-1]))
    return out


def plot_style() -> None:
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


def parse_bias_header(path: Path) -> dict[str, str]:
    with path.open() as handle:
        return dict(HEADER_RE.findall(handle.readline().strip()))


def load_bias_rows(path: Path) -> np.ndarray:
    rows = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#!")
                or stripped.startswith("!#")
                or stripped.startswith("time")
            ):
                continue
            parts = stripped.split()
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    return np.array(rows, dtype=float)


def gaussian_1d(grid: np.ndarray, centers: np.ndarray, sigmas: np.ndarray, heights: np.ndarray) -> np.ndarray:
    bias = np.zeros_like(grid)
    for center, sigma, height in zip(centers, sigmas, heights):
        bias += height * np.exp(-0.5 * ((grid - center) / sigma) ** 2)
    return -bias


def gaussian_2d(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    centers_x: np.ndarray,
    centers_y: np.ndarray,
    sigmas_x: np.ndarray,
    sigmas_y: np.ndarray,
    heights: np.ndarray,
) -> np.ndarray:
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="xy")
    bias = np.zeros_like(xx)
    for cx, cy, sx, sy, height in zip(centers_x, centers_y, sigmas_x, sigmas_y, heights):
        bias += height * np.exp(-0.5 * ((xx - cx) / sx) ** 2 - 0.5 * ((yy - cy) / sy) ** 2)
    return -bias


def analyze_umbrella_case(case_dir: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[UmbrellaWindow]] = defaultdict(list)
    for path in sorted((case_dir / "outputs").glob("mean_cv_energy_*.csv")):
        parsed = parse_mean_csv(path)
        if parsed is None:
            continue
        model_tag, window = parsed
        grouped[model_tag].append(window)

    generated: list[str] = []
    for model_tag, windows in sorted(grouped.items()):
        generated.extend(write_umbrella_outputs(model_tag, sorted(windows, key=lambda w: w.mean_cv), output_dir))
    return generated


def analyze_off24_case(case_dir: Path) -> list[str]:
    output_dir = case_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted((case_dir / "outputs").glob("mean_cv_energy_raccoon_off_off24_shift_*.csv"))
    windows = load_umbrella_windows(paths)
    return write_umbrella_outputs("off24_medium", windows, output_dir)


def analyze_bias_file(bias_path: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    header = parse_bias_header(bias_path)
    cv_number = int(header.get("CV_Number", "0"))
    rows = load_bias_rows(bias_path)
    if cv_number not in {1, 2} or len(rows) == 0:
        return []

    stem = bias_path.stem
    generated: list[str] = []
    if cv_number == 1:
        times = rows[:, 0]
        cv = rows[:, 1]
        sigma = rows[:, 2]
        heights = rows[:, 3]
        margin = max(3 * float(np.max(sigma)), 0.05)
        grid = np.linspace(float(np.min(cv) - margin), float(np.max(cv) + margin), 600)
        fes = gaussian_1d(grid, cv, sigma, heights)
        fes -= np.min(fes)

        csv_path = output_dir / f"{stem}_reconstructed_fes.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cv", "free_energy"])
            writer.writerows(zip(grid, fes))
        generated.append(str(csv_path))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 8.0), constrained_layout=True)
        ax1.plot(times, cv, color="#1f78b4")
        ax1.set_xlabel("Time (fs)")
        ax1.set_ylabel("CV (deg)")
        ax1.set_title(stem)
        ax1.grid(alpha=0.25)
        ax2.plot(grid, fes, color="#d95f02")
        ax2.set_xlabel("CV (deg)")
        ax2.set_ylabel("Relative free energy (arb. units)")
        ax2.grid(alpha=0.25)

        png_path = output_dir / f"{stem}_metad_plot.png"
        fig.savefig(png_path, dpi=300)
        plt.close(fig)
        generated.append(str(png_path))
    else:
        times = rows[:, 0]
        cv1 = rows[:, 1]
        cv2 = rows[:, 2]
        sigma1 = rows[:, 3]
        sigma2 = rows[:, 4]
        heights = rows[:, 5]
        margin1 = max(3 * float(np.max(sigma1)), 1.0)
        margin2 = max(3 * float(np.max(sigma2)), 1.0)
        grid1 = np.linspace(float(np.min(cv1) - margin1), float(np.max(cv1) + margin1), 180)
        grid2 = np.linspace(float(np.min(cv2) - margin2), float(np.max(cv2) + margin2), 180)
        fes = gaussian_2d(grid1, grid2, cv1, cv2, sigma1, sigma2, heights)
        fes -= np.min(fes)

        csv_path = output_dir / f"{stem}_reconstructed_fes.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cv1", "cv2", "free_energy"])
            for j, y in enumerate(grid2):
                for i, x in enumerate(grid1):
                    writer.writerow([x, y, fes[j, i]])
        generated.append(str(csv_path))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
        scatter = ax1.scatter(cv1, cv2, c=times, s=8, cmap="viridis")
        ax1.set_xlabel("CV1 (deg)")
        ax1.set_ylabel("CV2 (deg)")
        ax1.set_title(f"{stem} trajectory")
        fig.colorbar(scatter, ax=ax1, label="Time (fs)")

        contour = ax2.contourf(grid1, grid2, fes, levels=25, cmap="viridis")
        ax2.set_xlabel("CV1 (deg)")
        ax2.set_ylabel("CV2 (deg)")
        ax2.set_title(f"{stem} reconstructed FES")
        fig.colorbar(contour, ax=ax2, label="Relative free energy")

        png_path = output_dir / f"{stem}_metad_plot.png"
        fig.savefig(png_path, dpi=300)
        plt.close(fig)
        generated.append(str(png_path))
    return generated


def validate_umd_case(case_dir: Path) -> dict[str, object]:
    outputs_dir = case_dir / "outputs"
    expected_xyz = (
        case_dir / "optimized_malonaldehyde_initial.xyz"
        if case_dir.name == "malonaldehyd"
        else case_dir / "optimized_fmalonaldehyde_initial.xyz"
    )
    required = [
        case_dir / "run_U_MD.py",
        case_dir / "run_U_MDs_viper.sh",
        case_dir / "biASE.py",
        expected_xyz,
        outputs_dir,
        case_dir / "logs",
    ]
    missing = [str(path) for path in required if not path.exists()]
    counts = {
        "mean_cv_energy": len(list(outputs_dir.glob("mean_cv_energy_*.csv"))),
        "cv_energy": len(list(outputs_dir.glob("cv_energy_*.csv"))),
        "trajectories": len(list(outputs_dir.glob("umd_*.traj"))),
        "bfgs_logs": len(list(outputs_dir.glob("bfgs_*.log"))),
    }
    return {"case": str(case_dir), "missing": missing, "counts": counts}


def validate_azob_case(case_dir: Path) -> dict[str, object]:
    outputs_dir = case_dir / "outputs"
    required = [
        case_dir / "run_metad.py",
        case_dir / "run_metad_viper.sh",
        case_dir / "submit_chain.sh",
        case_dir / "config.env",
        case_dir / "logs",
        outputs_dir,
    ]
    start_files = [case_dir / "azob_cis_opt.traj", case_dir / "azob_trans_opt.traj"]
    missing = [str(path) for path in required if not path.exists()]
    missing.extend(str(path) for path in start_files if not path.exists())
    counts = {
        "bias_files": len(list(outputs_dir.glob("*.bias"))),
        "trajectories": len(list(outputs_dir.glob("*.traj"))),
        "bfgs_logs": len(list(outputs_dir.glob("bfgs_*.log"))),
    }
    return {"case": str(case_dir), "missing": missing, "counts": counts}


def create_combined_malon_plot(system_dir: Path) -> list[str]:
    analysis_dir = system_dir / "analysis"
    csv_sources = [
        analysis_dir,
        OUTPUT_ROOT / "upet" / system_dir.name,
        OUTPUT_ROOT / "sol3r" / system_dir.name,
    ]
    csv_paths: list[Path] = []
    for source_dir in csv_sources:
        csv_paths.extend(sorted(source_dir.glob("umbrella_integration_*.csv")))

    if not csv_paths:
        return []

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    generated: list[str] = []
    for csv_path in csv_paths:
        model_tag = csv_path.stem.replace("umbrella_integration_", "")
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        if data.size == 0:
            continue
        x = np.atleast_1d(data["mean_cv"])
        y = np.atleast_1d(data["free_energy"])
        color = MODEL_COLORS.get(model_tag, None)
        label = MODEL_LABELS.get(model_tag, model_tag.replace("_", " "))
        ax.plot(x, y, label=label, color=color)

    ax.set_xlabel("CV (A)")
    ax.set_ylabel("Relative free energy (arb. units)")
    ax.set_title(f"{system_dir.name}: umbrella integration comparison")
    ax.grid(alpha=0.22)
    ax.legend(loc="best", frameon=False, ncol=1)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = analysis_dir / f"{system_dir.name}_combined_energy_models.{ext}"
        fig.savefig(out, dpi=300)
        generated.append(str(out))
    plt.close(fig)
    return generated


def analyze_additional_systems() -> dict[str, object]:
    plot_style()
    OUTPUT_ROOT.mkdir(exist_ok=True)
    summary: dict[str, object] = {"systems": {}, "combined_plots": []}

    for model_kind, model_root in SYSTEM_ROOTS.items():
        model_summary: dict[str, object] = {"processed": [], "missing": []}

        for case_name in UMD_CASES:
            case_dir = model_root / case_name
            validation = validate_umd_case(case_dir)
            if validation["missing"]:
                model_summary["missing"].append(validation)
            output_dir = OUTPUT_ROOT / model_kind / case_name
            generated = analyze_umbrella_case(case_dir, output_dir)
            model_summary["processed"].append(
                {"case": str(case_dir), "generated": generated, "counts": validation["counts"]}
            )

        for case_name in AZOB_CASES:
            case_dir = model_root / "azobenzene" / case_name
            validation = validate_azob_case(case_dir)
            if validation["missing"]:
                model_summary["missing"].append(validation)
            output_dir = OUTPUT_ROOT / model_kind / "azobenzene" / case_name
            generated: list[str] = []
            for bias_path in sorted((case_dir / "outputs").glob("*.bias")):
                generated.extend(analyze_bias_file(bias_path, output_dir))
            model_summary["processed"].append(
                {"case": str(case_dir), "generated": generated, "counts": validation["counts"]}
            )

        summary["systems"][model_kind] = model_summary

    for system_name in ("malonaldehyd", "f-malonaldehyd"):
        summary["systems"].setdefault("raccoon_off24", {})[system_name] = analyze_off24_case(REPO_ROOT / system_name)
        summary["combined_plots"].extend(create_combined_malon_plot(REPO_ROOT / system_name))

    return summary


def write_summary(summary: dict[str, object]) -> Path:
    summary_path = OUTPUT_ROOT / "analysis_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def main() -> None:
    summary = analyze_additional_systems()
    summary_path = write_summary(summary)
    print(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
