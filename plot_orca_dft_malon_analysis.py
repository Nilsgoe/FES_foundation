from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import extend_malon_analysis as ui
import plot_malon_umbrella_integration as diff_ui
from plot_malon_gpr_analysis import run_upstream_analysis


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
DEFAULT_RAW = ROOT / "analysis/orca_dft/malonaldehyd/raw"
DEFAULT_OUT = ROOT / "analysis/orca_dft/malonaldehyd"
WINDOW_RE = re.compile(r"window_(\d+)_shift_([+-]\d+)_cv_energy\.csv$")
MODEL_COLOR = "#2b6f8a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORCA DFT umbrella analysis using the vendored GitHub workflow.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-tag", default="orca_dft")
    parser.add_argument("--model-label", default="ORCA wB97M-V/def2-TZVPD")
    parser.add_argument(
        "--burn-fraction",
        type=float,
        default=0.0,
        help="Discard this fraction from the start of each window before computing means.",
    )
    return parser.parse_args()


def read_windows(path: Path) -> dict[int, dict[str, float | int]]:
    with path.open() as handle:
        return {
            int(row["window_id"]): {
                "window_id": int(row["window_id"]),
                "shift": int(row["shift"]),
                "center": float(row["center_A"]),
                "kappa": float(row["kappa_eV_A2"]),
            }
            for row in csv.DictReader(handle)
        }


def load_series(path: Path, burn_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    data = np.atleast_1d(data)
    cv = np.asarray(data["cv_A"], dtype=float)
    energy = np.asarray(data["potential_eV"], dtype=float)
    finite = np.isfinite(cv) & np.isfinite(energy)
    cv = cv[finite]
    energy = energy[finite]
    if not 0.0 <= burn_fraction < 1.0:
        raise ValueError("--burn-fraction must be in [0, 1).")
    start = int(np.floor(len(cv) * burn_fraction))
    return cv[start:], energy[start:]


def gather_windows(raw_dir: Path, burn_fraction: float, model_tag: str) -> list[ui.WindowStats]:
    window_defs = read_windows(raw_dir / "windows.csv")
    windows: list[ui.WindowStats] = []
    for path in sorted(raw_dir.glob("window_*_cv_energy.csv")):
        match = WINDOW_RE.match(path.name)
        if not match:
            continue
        window_id = int(match.group(1))
        definition = window_defs[window_id]
        center = float(definition["center"])
        kappa = float(definition["kappa"])
        shift = int(definition["shift"])
        cv, energy = load_series(path, burn_fraction)
        if len(cv) < 8:
            continue
        force_samples = kappa * (center - cv)
        tau_int, force_stderr = ui.estimate_mean_stderr(force_samples)
        n_eff = max(len(force_samples) / (2.0 * tau_int), 1.0)
        windows.append(
            ui.WindowStats(
                model_tag=model_tag,
                shift=shift,
                center=center,
                mean_cv=float(np.mean(cv)),
                mean_force=float(np.mean(force_samples)),
                mean_energy=float(np.mean(energy)),
                initial_cv=float(cv[0]),
                cv_var=float(np.var(cv, ddof=1)),
                cv_std=float(np.std(cv, ddof=1)),
                n_samples=int(len(cv)),
                tau_int=float(tau_int),
                n_eff=float(n_eff),
                force_stderr=float(force_stderr),
                mean_path=path,
                raw_path=path,
            )
        )
    return sorted(windows, key=lambda window: window.center)


def write_mace_style_outputs(
    output_dir: Path, raw_dir: Path, windows: list[ui.WindowStats], model_tag: str
) -> Path:
    """Convert ORCA window CSVs into the exact layout used by the MACE UMD analysis."""
    dataset_dir = output_dir / "mace_style_dataset"
    outputs_dir = dataset_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    diff_gp_link = dataset_dir / "diff_gp.py"
    if not diff_gp_link.exists():
        diff_gp_link.symlink_to(ROOT / "malonaldehyd" / "diff_gp.py")

    for window in windows:
        raw_data = np.genfromtxt(raw_dir / window.raw_path.name, delimiter=",", names=True)
        raw_data = np.atleast_1d(raw_data)
        raw_out = outputs_dir / f"cv_energy_{model_tag}_shift_{window.shift}.csv"
        mean_out = outputs_dir / f"mean_cv_energy_{model_tag}_shift_{window.shift}.csv"
        with raw_out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cv", "energy"])
            for cv, energy in zip(raw_data["cv_A"], raw_data["potential_eV"]):
                if np.isfinite(cv) and np.isfinite(energy):
                    writer.writerow([float(cv), float(energy)])
        with mean_out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    window.center,
                    window.mean_cv,
                    window.mean_force,
                    window.mean_energy,
                    window.initial_cv,
                ]
            )
    return dataset_dir


def gpr_to_shared_plot_result(results: dict[str, np.ndarray | float], windows: list[ui.WindowStats]) -> dict[str, np.ndarray | float]:
    """Expose the SciPy GPR result with the same keys as the diff_GP cache used by MACE plots."""
    return {
        **results,
        "x_train": np.array([w.center for w in windows], dtype=float),
        "dy_train": np.array([w.mean_force for w in windows], dtype=float),
        "x_predict": np.asarray(results["x_star"], dtype=float),
        "pmf": np.asarray(results["pmf_mean"], dtype=float),
        "mean_force": np.asarray(results["deriv_mean"], dtype=float),
        "mean_force_std": np.asarray(results["deriv_std"], dtype=float),
        "sigma": np.nan,
        "delta": np.nan,
        "lengthscale": float(results["lengthscale"]),
        "alpha_RQ": np.nan,
    }


def as_upstream_windows(windows: list[ui.WindowStats]) -> list[diff_ui.Window]:
    return [
        diff_ui.Window(
            model_tag=window.model_tag,
            shift=window.shift,
            center=window.center,
            mean_cv=window.mean_cv,
            mean_force=window.mean_force,
            raw_path=window.raw_path,
        )
        for window in windows
    ]


def write_window_summary(output_dir: Path, windows: list[ui.WindowStats]) -> Path:
    out = output_dir / "orca_dft_window_summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shift",
                "center_A",
                "mean_cv_A",
                "mean_force_eV_A",
                "mean_energy_eV",
                "cv_std_A",
                "n_samples",
                "tau_int",
                "n_eff",
                "force_stderr_eV_A",
                "source_file",
            ]
        )
        for window in windows:
            writer.writerow(
                [
                    window.shift,
                    window.center,
                    window.mean_cv,
                    window.mean_force,
                    window.mean_energy,
                    window.cv_std,
                    window.n_samples,
                    window.tau_int,
                    window.n_eff,
                    window.force_stderr,
                    window.raw_path.name,
                ]
            )
    return out


def main() -> None:
    args = parse_args()
    ui.plot_style()
    ui.MODEL_LABELS[args.model_tag] = args.model_label
    ui.MODEL_COLORS[args.model_tag] = MODEL_COLOR
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_windows = gather_windows(args.raw_dir, args.burn_fraction, args.model_tag)
    if len(source_windows) < 3:
        raise RuntimeError(f"Need at least 3 windows, found {len(source_windows)} in {args.raw_dir}")
    dataset_dir = write_mace_style_outputs(args.output_dir, args.raw_dir, source_windows, args.model_tag)
    grouped = ui.gather_windows(dataset_dir)
    windows = grouped.get(args.model_tag, [])
    if len(windows) < 3:
        raise RuntimeError(f"Converted MACE-style dataset has only {len(windows)} windows.")
    results = gpr_to_shared_plot_result(ui.run_gpr_from_windows(windows), windows)
    outputs = [write_window_summary(args.output_dir, windows)]
    outputs.extend(ui.save_ui_outputs(args.output_dir, dataset_dir, args.model_tag, windows, results))
    outputs.extend(ui.save_gpr_outputs(args.output_dir, dataset_dir, args.model_tag, windows, results))
    outputs.extend(run_upstream_analysis(args.output_dir, args.model_tag, as_upstream_windows(windows)))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
