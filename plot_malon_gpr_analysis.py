from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_malon_umbrella_integration as ui


UPSTREAM_ROOT = ui.ROOT / "external" / "GPR-Umbrella-Sampling-Analysis"
sys.path.insert(0, str(UPSTREAM_ROOT))

from gpr_umbrella_1d import gpr_umbrella_integration, plot_diagnostics  # noqa: E402


KJ_PER_MOL_PER_EV = 96.485


def read_positions(raw_path: Path) -> np.ndarray:
    data = np.genfromtxt(raw_path, delimiter=",", names=True)
    names = data.dtype.names or ()
    if "cv" not in names:
        raise ValueError(f"No 'cv' column found in {raw_path}")
    return np.atleast_1d(data["cv"]).astype(float)


def write_upstream_inputs(windows: list[ui.Window], input_dir: Path) -> list[Path]:
    written: list[Path] = []
    for idx, window in enumerate(sorted(windows, key=lambda w: w.shift)):
        raw_path = window.raw_path
        if not raw_path.exists() and window.model_tag == "viper_upet_pet_spice_rot":
            candidate = raw_path.with_name(f"cv_energy_pet_spice_rotavg3_shift_{window.shift}.csv")
            if candidate.exists():
                raw_path = candidate
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw trajectory CSV for {window.model_tag} shift {window.shift}: {window.raw_path}")
        positions = read_positions(raw_path)
        # The upstream loader converts kJ/mol to eV for data_folder inputs, so
        # write kappa in kJ/mol/A^2 to preserve our physical KAPPA=50 eV/A^2.
        kappa_kj_mol = ui.KAPPA * KJ_PER_MOL_PER_EV
        rows = np.column_stack(
            [
                np.arange(len(positions), dtype=float),
                positions,
                np.full(len(positions), window.center, dtype=float),
                np.full(len(positions), kappa_kj_mol, dtype=float),
            ]
        )
        path = input_dir / f"window_{idx:03d}.ui_dat"
        np.savetxt(path, rows, fmt=["%.0f", "%.10f", "%.10f", "%.10f"])
        written.append(path)
    return written


def run_upstream_analysis(output_dir: Path, model_tag: str, windows: list[ui.Window]) -> list[Path]:
    label = ui.MODEL_LABELS.get(model_tag, model_tag.replace("_", " "))
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{model_tag}_gpr_ui_") as tmp:
        input_dir = Path(tmp)
        write_upstream_inputs(windows, input_dir)
        results = gpr_umbrella_integration(
            data_folder=str(input_dir),
            cv_unit="A",
            energy_unit="eV",
            output_prefix=f"{output_dir.parent.name}: {label}",
            output_dir=str(output_dir),
            n_star=200,
            plot=False,
            save_fig=False,
            save_outputs=False,
            verbose=False,
        )

    fig = plot_diagnostics(results, output_prefix=f"{output_dir.parent.name}: {label}")
    outputs: list[Path] = []
    for ext in ("png", "pdf"):
        path = output_dir / f"umbrella_integration_{model_tag}_analysis.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the vendored GPR-Umbrella-Sampling-Analysis diagnostics for malonaldehyde umbrella data."
    )
    parser.add_argument(
        "--only",
        metavar="TEXT",
        help="Only process targets whose 'system:model' string contains TEXT, e.g. 'malonaldehyd:off_large'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many targets. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--missing-ok",
        action="store_true",
        help="Report missing raw CSV files and continue instead of failing immediately.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing: list[str] = []
    processed = 0
    for system_name, _system_dir, model_tag, windows, output_dir in ui.iter_targets():
        target_name = f"{system_name}:{model_tag}"
        if args.only and args.only not in target_name:
            continue
        if args.limit is not None and processed >= args.limit:
            break
        try:
            print(f"Running GitHub GPR-Umbrella analysis for {target_name}", flush=True)
            for path in run_upstream_analysis(output_dir, model_tag, windows):
                print(path, flush=True)
            processed += 1
        except FileNotFoundError as exc:
            if not args.missing_ok:
                raise
            missing.append(str(exc))

    if missing:
        print("\nMissing inputs:")
        for item in missing:
            print(f"- {item}")


if __name__ == "__main__":
    main()
