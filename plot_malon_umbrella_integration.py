from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
SYSTEMS = ("malonaldehyd", "f-malonaldehyd")
KAPPA = 50.0

MEAN_RE = re.compile(r"mean_cv_energy_(.+?)_shift_(-?\d+)(?:_.+)?\.csv$")
RAW_RE = re.compile(r"cv_energy_(.+?)_shift_(-?\d+)(?:_.+)?\.csv$")

VIPER_TARGETS = {
    "malonaldehyd": (
        (
            "viper_sol3r",
            ROOT / "viper_analysis/sol3r/malonaldehyd/umbrella_integration_viper_sol3r.csv",
            Path("/work/gpuviper_ptmp/Enhanced_sampling/sol3r/malonaldehyd/outputs"),
        ),
        (
            "viper_upet_pet_spice",
            ROOT / "viper_analysis/upet/malonaldehyd/umbrella_integration_viper_upet_pet_spice.csv",
            Path("/work/gpuviper_ptmp/Enhanced_sampling/upet/malonaldehyd/outputs"),
        ),
        (
            "viper_upet_pet_spice_rot",
            ROOT
            / "viper_analysis/upet/malonaldehyd_pet_spice_rotavg3_43w/umbrella_integration_viper_upet_pet_spice_rot.csv",
            ROOT / "viper_analysis/upet/malonaldehyd_pet_spice_rotavg3_43w/outputs",
        ),
    ),
    "f-malonaldehyd": (
        (
            "viper_sol3r",
            ROOT / "viper_analysis/sol3r/f-malonaldehyd/umbrella_integration_viper_sol3r.csv",
            Path("/work/gpuviper_ptmp/Enhanced_sampling/sol3r/f-malonaldehyd/outputs"),
        ),
        (
            "viper_upet_pet_spice",
            ROOT / "viper_analysis/upet/f-malonaldehyd/umbrella_integration_viper_upet_pet_spice.csv",
            Path("/work/gpuviper_ptmp/Enhanced_sampling/upet/f-malonaldehyd/outputs"),
        ),
        (
            "viper_upet_pet_spice_rot",
            ROOT
            / "viper_analysis/upet/f-malonaldehyd_pet_spice_rotavg3_43w/umbrella_integration_viper_upet_pet_spice_rot.csv",
            ROOT / "viper_analysis/upet/f-malonaldehyd_pet_spice_rotavg3_43w/outputs",
        ),
    ),
}

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
    "viper_sol3r": "SO3LR",
    "viper_upet_pet_spice": "PET-SPICE",
    "viper_upet_pet_spice_rot": "PET-SPICE_rot",
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
    "viper_sol3r": "#6a3d9a",
    "viper_upet_pet_spice": "#e31a1c",
    "viper_upet_pet_spice_rot": "#ff7f00",
}


@dataclass(frozen=True)
class Window:
    model_tag: str
    shift: int
    center: float
    mean_cv: float
    mean_force: float
    raw_path: Path


def normalize_model_tag(raw: str) -> str:
    if raw.startswith("raccoon_"):
        raw = raw[len("raccoon_") :]
    return {"off_off24": "off24_medium"}.get(raw, raw)


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
            "lines.linewidth": 2.3,
            "savefig.bbox": "tight",
        }
    )


def load_diff_gp(system_dir: Path):
    module_path = system_dir / "diff_gp.py"
    module_name = f"diff_gp_{system_dir.name.replace('-', '_')}"
    if module_name in sys.modules:
        return sys.modules[module_name].diff_GP
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load diff_GP from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.diff_GP


def local_windows(system_dir: Path) -> dict[str, list[Window]]:
    mean_files: dict[tuple[str, int], Path] = {}
    raw_files: dict[tuple[str, int], Path] = {}
    for path in (system_dir / "outputs").glob("mean_cv_energy_*.csv"):
        match = MEAN_RE.match(path.name)
        if match:
            mean_files[(normalize_model_tag(match.group(1)), int(match.group(2)))] = path
    for path in (system_dir / "outputs").glob("cv_energy_*.csv"):
        match = RAW_RE.match(path.name)
        if match:
            raw_files[(normalize_model_tag(match.group(1)), int(match.group(2)))] = path

    grouped: dict[str, list[Window]] = {}
    for key, mean_path in sorted(mean_files.items()):
        raw_path = raw_files.get(key)
        if raw_path is None:
            continue
        model_tag, shift = key
        with mean_path.open() as handle:
            center, mean_cv, mean_force, *_ = [float(x) for x in next(csv.reader(handle))[:5]]
        grouped.setdefault(model_tag, []).append(Window(model_tag, shift, center, mean_cv, mean_force, raw_path))
    return {model: sorted(windows, key=lambda w: w.shift) for model, windows in grouped.items()}


def viper_windows(model_tag: str, summary_csv: Path, raw_dir: Path) -> list[Window]:
    data = np.atleast_1d(np.genfromtxt(summary_csv, delimiter=",", names=True, dtype=None, encoding=None))
    windows: list[Window] = []
    for row in data:
        shift = int(row["shift"])
        raw_path = raw_dir / f"cv_energy_{model_tag}_shift_{shift}.csv"
        windows.append(
            Window(
                model_tag=model_tag,
                shift=shift,
                center=float(row["window_center"]),
                mean_cv=float(row["mean_cv"]),
                mean_force=float(row["mean_force"]),
                raw_path=raw_path,
            )
        )
    return sorted(windows, key=lambda w: w.shift)


def cache_paths(output_dir: Path, model_tag: str) -> tuple[Path, Path]:
    return output_dir / f"umbrella_gpr_{model_tag}_pmf.csv", output_dir / f"umbrella_gpr_{model_tag}_deriv.csv"


def load_cache(output_dir: Path, model_tag: str) -> dict[str, np.ndarray] | None:
    pmf_csv, deriv_csv = cache_paths(output_dir, model_tag)
    if not pmf_csv.exists() or not deriv_csv.exists():
        return None
    pmf_data = np.atleast_1d(np.genfromtxt(pmf_csv, delimiter=",", names=True))
    deriv_data = np.atleast_1d(np.genfromtxt(deriv_csv, delimiter=",", names=True))
    if pmf_data.size == 0 or deriv_data.size == 0:
        return None
    pmf_names = pmf_data.dtype.names or ()
    deriv_names = deriv_data.dtype.names or ()
    if "cv" not in pmf_names or "cv" not in deriv_names:
        return None
    pmf_col = "pmf" if "pmf" in pmf_names else "pmf_mean"
    force_col = "mean_force" if "mean_force" in deriv_names else "deriv_mean"
    std_col = "mean_force_std" if "mean_force_std" in deriv_names else "deriv_std"
    if pmf_col not in pmf_names or force_col not in deriv_names or std_col not in deriv_names:
        return None
    return {
        "x": np.asarray(pmf_data["cv"], dtype=float),
        "pmf": np.asarray(pmf_data[pmf_col], dtype=float),
        "force_x": np.asarray(deriv_data["cv"], dtype=float),
        "force": np.asarray(deriv_data[force_col], dtype=float),
        "force_std": np.asarray(deriv_data[std_col], dtype=float),
    }


def fit_diff_gp(system_dir: Path, windows: list[Window]) -> dict[str, np.ndarray]:
    diff_gp_cls = load_diff_gp(system_dir)
    shifts = np.array([w.shift for w in windows], dtype=float)
    x_train = np.array([w.center for w in windows], dtype=float)
    dy_train = np.array([w.mean_force for w in windows], dtype=float)
    initial_cv = float(np.mean([w.center - 0.05 * w.shift for w in windows]))
    x_predict = initial_cv + 0.05 * np.linspace(shifts.min(), shifts.max(), len(windows))

    gp = diff_gp_cls(verbose=False, learning_rate=1e-5, momentum=0.5)
    gp.optimize(x_train, dy_train)
    gp.train(x_train, dy_train)
    pmf, _ = gp.predict(x_predict)
    force, force_std = gp.predict_diff(x_predict)
    pmf = np.asarray(pmf, dtype=float)
    pmf -= np.nanmin(pmf)
    return {
        "x": x_predict,
        "pmf": pmf,
        "force_x": x_predict,
        "force": np.asarray(force, dtype=float),
        "force_std": np.asarray(force_std, dtype=float),
    }


def save_cache(output_dir: Path, model_tag: str, results: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pmf_csv, deriv_csv = cache_paths(output_dir, model_tag)
    np.savetxt(
        pmf_csv,
        np.column_stack([results["x"], results["pmf"]]),
        delimiter=",",
        header="cv,pmf",
        comments="",
    )
    np.savetxt(
        deriv_csv,
        np.column_stack([results["force_x"], results["force"], results["force_std"]]),
        delimiter=",",
        header="cv,mean_force,mean_force_std",
        comments="",
    )


def plot_diff_gp(output_dir: Path, model_tag: str, windows: list[Window], results: dict[str, np.ndarray]) -> list[Path]:
    color = MODEL_COLORS.get(model_tag, "#1f78b4")
    label = MODEL_LABELS.get(model_tag, model_tag.replace("_", " "))
    x_windows = np.array([w.center for w in windows], dtype=float)
    y_windows = np.array([w.mean_force for w in windows], dtype=float)

    fig, ax_pmf = plt.subplots(figsize=(8.0, 5.2))
    ax_pmf.plot(
        results["x"],
        results["pmf"] - np.nanmin(results["pmf"]),
        color=color,
        label=f"{label} diff_GP PMF",
    )
    ax_pmf.set_xlabel("CV (A)")
    ax_pmf.set_ylabel("Relative free energy (eV)")
    ax_pmf.set_title(f"{output_dir.parent.name}: {label}")
    ax_pmf.grid(alpha=0.25)

    ax_force = ax_pmf.twinx()
    ax_force.scatter(x_windows, y_windows, s=28, color="#333333", alpha=0.75, label="Window mean force")
    ax_force.plot(results["force_x"], results["force"], color="#111111", linestyle="--", label="diff_GP mean force")
    ax_force.fill_between(
        results["force_x"],
        results["force"] - 2.0 * results["force_std"],
        results["force"] + 2.0 * results["force_std"],
        color="#777777",
        alpha=0.20,
        linewidth=0,
        label="Mean force ±2σ",
    )
    ax_force.set_ylabel("Mean force (eV/A)")

    lines_pmf, labels_pmf = ax_pmf.get_legend_handles_labels()
    lines_force, labels_force = ax_force.get_legend_handles_labels()
    ax_pmf.legend(lines_pmf + lines_force, labels_pmf + labels_force, loc="best", frameon=False)
    fig.tight_layout()

    outputs: list[Path] = []
    for ext in ("png", "pdf"):
        out = output_dir / f"umbrella_integration_{model_tag}.{ext}"
        fig.savefig(out, dpi=300)
        outputs.append(out)
    plt.close(fig)
    return outputs


def iter_targets() -> list[tuple[str, Path, str, list[Window], Path]]:
    targets: list[tuple[str, Path, str, list[Window], Path]] = []
    for system_name in SYSTEMS:
        system_dir = ROOT / system_name
        for model_tag, windows in local_windows(system_dir).items():
            targets.append((system_name, system_dir, model_tag, windows, system_dir / "analysis"))
        for model_tag, summary_csv, raw_dir in VIPER_TARGETS[system_name]:
            if summary_csv.exists():
                targets.append((system_name, system_dir, model_tag, viper_windows(model_tag, summary_csv, raw_dir), summary_csv.parent))
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot malonaldehyde umbrella-integration PMFs with diff_GP mean-force uncertainty."
    )
    parser.add_argument("--refit", action="store_true", help="Refit diff_GP instead of reusing umbrella_gpr_* CSV caches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_style()
    for _system_name, system_dir, model_tag, windows, output_dir in iter_targets():
        if not windows:
            continue
        results = None if args.refit else load_cache(output_dir, model_tag)
        if results is None:
            results = fit_diff_gp(system_dir, windows)
            save_cache(output_dir, model_tag, results)
        for path in plot_diff_gp(output_dir, model_tag, windows, results):
            print(path)


if __name__ == "__main__":
    main()
