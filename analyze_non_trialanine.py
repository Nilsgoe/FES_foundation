from __future__ import annotations

import csv
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


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
TARGET_DIRS = [
    ROOT / "azobenzene",
    ROOT / "malonaldehyd",
    ROOT / "f-malonaldehyd",
]

MEAN_CSV_RE = re.compile(r"mean_cv_energy_raccoon_(.+)_shift_(-?\d+)_.*\.csv$")
METAD_TXT_RE = re.compile(r"metad.*\.txt$")
HEADER_RE = re.compile(r"([A-Za-z_]+)=([A-Za-z0-9.+-]+)")


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
    match = MEAN_CSV_RE.match(path.name)
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


def cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    if len(x) < 2:
        return out
    dx = np.diff(x)
    avg = 0.5 * (y[1:] + y[:-1])
    out[1:] = np.cumsum(dx * avg)
    return out


def analyze_umbrella(case_dir: Path) -> list[Path]:
    outputs_dir = case_dir / "outputs"
    analysis_dir = case_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    grouped: dict[str, list[UmbrellaWindow]] = defaultdict(list)
    for path in sorted(outputs_dir.glob("mean_cv_energy_*.csv")):
        parsed = parse_mean_csv(path)
        if parsed is None:
            continue
        model_tag, window = parsed
        grouped[model_tag].append(window)

    generated: list[Path] = []
    for model_tag, windows in sorted(grouped.items()):
        windows = sorted(windows, key=lambda w: w.mean_cv)
        x = np.array([w.mean_cv for w in windows], dtype=float)
        force = np.array([w.mean_force for w in windows], dtype=float)
        centers = np.array([w.center for w in windows], dtype=float)
        free_energy = cumulative_trapezoid(x, force)
        free_energy -= np.min(free_energy)

        csv_path = analysis_dir / f"umbrella_integration_{model_tag}.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["mean_cv", "free_energy", "mean_force", "window_center", "shift", "source_file"]
            )
            for window, xi, fi, dfi in zip(windows, x, free_energy, force):
                writer.writerow([xi, fi, dfi, window.center, window.shift, window.source.name])
        generated.append(csv_path)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(x, free_energy, marker="o", linewidth=2.5, label="Umbrella integration FES")
        ax1.set_xlabel("CV")
        ax1.set_ylabel("Free energy (arb. units)")
        ax1.set_title(f"{case_dir.name}: {model_tag}")
        ax1.grid(alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(x, force, color="tab:red", linestyle="--", alpha=0.8, label="Mean force")
        ax2.scatter(centers, force, color="tab:red", s=20, alpha=0.8)
        ax2.set_ylabel("Mean force")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        fig.tight_layout()

        png_path = analysis_dir / f"umbrella_integration_{model_tag}.png"
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        generated.append(png_path)

    return generated


def parse_metad_header(path: Path) -> dict[str, str]:
    first_line = path.open().readline().strip()
    return dict(HEADER_RE.findall(first_line))


def load_metad_rows(path: Path) -> np.ndarray:
    rows = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#!") or stripped.startswith("!#") or "time" in stripped:
                continue
            parts = stripped.split()
            rows.append([float(x) for x in parts])
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


def meta_analysis_dir(txt_path: Path) -> Path:
    analysis_dir = txt_path.parent.parent / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    return analysis_dir


def analyze_metad_1d(txt_path: Path, rows: np.ndarray) -> list[Path]:
    analysis_dir = meta_analysis_dir(txt_path)
    times = rows[:, 0]
    cv = rows[:, 1]
    sigma = rows[:, 2]
    heights = rows[:, 3]

    margin = max(3 * float(np.max(sigma)), 0.05)
    grid = np.linspace(float(np.min(cv) - margin), float(np.max(cv) + margin), 600)
    fes = gaussian_1d(grid, cv, sigma, heights)
    fes -= np.min(fes)

    stem = txt_path.stem
    csv_path = analysis_dir / f"{stem}_reconstructed_fes.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "free_energy"])
        writer.writerows(zip(grid, fes))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    ax1.plot(times, cv, linewidth=1.5)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("CV")
    ax1.set_title(stem)
    ax1.grid(alpha=0.25)

    ax2.plot(grid, fes, linewidth=2.5)
    ax2.set_xlabel("CV")
    ax2.set_ylabel("Reconstructed free energy")
    ax2.grid(alpha=0.25)

    png_path = analysis_dir / f"{stem}_metad_plot.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return [csv_path, png_path]


def analyze_metad_2d(txt_path: Path, rows: np.ndarray) -> list[Path]:
    analysis_dir = meta_analysis_dir(txt_path)
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

    stem = txt_path.stem
    csv_path = analysis_dir / f"{stem}_reconstructed_fes.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv1", "cv2", "free_energy"])
        for j, y in enumerate(grid2):
            for i, x in enumerate(grid1):
                writer.writerow([x, y, fes[j, i]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    scatter = ax1.scatter(cv1, cv2, c=times, s=8, cmap="viridis")
    ax1.set_xlabel("CV1")
    ax1.set_ylabel("CV2")
    ax1.set_title(f"{stem} trajectory")
    fig.colorbar(scatter, ax=ax1, label="Time")

    contour = ax2.contourf(grid1, grid2, fes, levels=25, cmap="viridis")
    ax2.set_xlabel("CV1")
    ax2.set_ylabel("CV2")
    ax2.set_title(f"{stem} reconstructed FES")
    fig.colorbar(contour, ax=ax2, label="Free energy")

    png_path = analysis_dir / f"{stem}_metad_plot.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return [csv_path, png_path]


def analyze_metad(case_dir: Path) -> list[Path]:
    generated: list[Path] = []
    for txt_path in sorted(case_dir.rglob("*.txt")):
        if "analysis" in txt_path.parts or "logs" in txt_path.parts:
            continue
        if not METAD_TXT_RE.match(txt_path.name):
            continue
        header = parse_metad_header(txt_path)
        cv_number = int(header.get("CV_Number", "0"))
        if cv_number not in {1, 2}:
            continue
        rows = load_metad_rows(txt_path)
        if len(rows) == 0:
            continue
        if cv_number == 1:
            generated.extend(analyze_metad_1d(txt_path, rows))
        else:
            generated.extend(analyze_metad_2d(txt_path, rows))
    return generated


def main() -> None:
    all_generated: dict[str, list[Path]] = {}
    for case_dir in TARGET_DIRS:
        generated: list[Path] = []
        if case_dir.name in {"malonaldehyd", "f-malonaldehyd"}:
            generated.extend(analyze_umbrella(case_dir))
        generated.extend(analyze_metad(case_dir))
        all_generated[case_dir.name] = generated

    for case_name, paths in all_generated.items():
        print(f"[{case_name}] generated {len(paths)} files")
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
