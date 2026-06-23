from __future__ import annotations

import argparse
from pathlib import Path

import extend_malon_analysis as ema


ROOT = ema.ROOT

TARGETS = (
    (
        "malonaldehyd",
        "viper_sol3r",
        "viper_sol3r",
        Path("/work/gpuviper_ptmp/Enhanced_sampling/sol3r/malonaldehyd"),
        ROOT / "viper_analysis" / "sol3r" / "malonaldehyd",
        -7,
        35,
    ),
    (
        "f-malonaldehyd",
        "viper_sol3r",
        "viper_sol3r",
        Path("/work/gpuviper_ptmp/Enhanced_sampling/sol3r/f-malonaldehyd"),
        ROOT / "viper_analysis" / "sol3r" / "f-malonaldehyd",
        -7,
        39,
    ),
    (
        "malonaldehyd",
        "viper_upet_pet_spice",
        "viper_upet_pet_spice",
        Path("/work/gpuviper_ptmp/Enhanced_sampling/upet/malonaldehyd"),
        ROOT / "viper_analysis" / "upet" / "malonaldehyd",
        -7,
        35,
    ),
    (
        "f-malonaldehyd",
        "viper_upet_pet_spice",
        "viper_upet_pet_spice",
        Path("/work/gpuviper_ptmp/Enhanced_sampling/upet/f-malonaldehyd"),
        ROOT / "viper_analysis" / "upet" / "f-malonaldehyd",
        -7,
        39,
    ),
    (
        "malonaldehyd",
        "pet_spice_rotavg3",
        "viper_upet_pet_spice_rot",
        ROOT / "viper_analysis" / "upet" / "malonaldehyd_pet_spice_rotavg3_43w",
        ROOT / "viper_analysis" / "upet" / "malonaldehyd_pet_spice_rotavg3_43w",
        -7,
        35,
    ),
    (
        "f-malonaldehyd",
        "pet_spice_rotavg3",
        "viper_upet_pet_spice_rot",
        ROOT / "viper_analysis" / "upet" / "f-malonaldehyd_pet_spice_rotavg3_43w",
        ROOT / "viper_analysis" / "upet" / "f-malonaldehyd_pet_spice_rotavg3_43w",
        -7,
        39,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate Viper malonaldehyde umbrella analyses from raw CSV files.")
    parser.add_argument("--system", choices=("malonaldehyd", "f-malonaldehyd"), help="Limit to one system.")
    parser.add_argument(
        "--model",
        choices=("viper_sol3r", "viper_upet_pet_spice", "viper_upet_pet_spice_rot"),
        help="Limit to one output model tag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ema.plot_style()
    for system_name, source_tag, output_tag, raw_parent, output_dir, first_shift, last_shift in TARGETS:
        if args.system and system_name != args.system:
            continue
        if args.model and output_tag != args.model:
            continue
        print(f"Regenerating {system_name} {output_tag} from {raw_parent / 'outputs'}", flush=True)
        source_dir = ROOT / system_name
        grouped = ema.gather_windows(raw_parent)
        if source_tag not in grouped:
            raise RuntimeError(f"No {source_tag} windows found under {raw_parent / 'outputs'}")

        windows = grouped[source_tag]
        shifts = sorted(window.shift for window in windows)
        expected = list(range(first_shift, last_shift + 1))
        if shifts != expected:
            raise RuntimeError(
                f"{raw_parent}: expected shifts {first_shift}..{last_shift}, "
                f"got {shifts[:5]} ... {shifts[-5:]}"
            )
        print(f"  windows: {len(windows)} shifts {shifts[0]}..{shifts[-1]}", flush=True)

        diff_gp_results = ema.run_diff_gp_from_windows(source_dir, windows)
        for path in ema.save_ui_outputs(output_dir, source_dir, output_tag, windows, diff_gp_results):
            print(path)
        for path in ema.save_gpr_outputs(output_dir, source_dir, output_tag, windows, diff_gp_results):
            print(path)


if __name__ == "__main__":
    main()
