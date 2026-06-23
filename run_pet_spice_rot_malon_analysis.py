from __future__ import annotations

from pathlib import Path

import extend_malon_analysis as ema


ROOT = ema.ROOT
TARGETS = (
    (
        "malonaldehyd",
        ROOT / "viper_analysis" / "upet" / "malonaldehyd_pet_spice_rotavg3_43w",
        -7,
        35,
    ),
    (
        "f-malonaldehyd",
        ROOT / "viper_analysis" / "upet" / "f-malonaldehyd_pet_spice_rotavg3_43w",
        -7,
        39,
    ),
)
MODEL_TAG = "viper_upet_pet_spice_rot"


def main() -> None:
    ema.plot_style()
    for system_name, target_dir, first_shift, last_shift in TARGETS:
        source_dir = ROOT / system_name
        grouped = ema.gather_windows(target_dir)
        if "pet_spice_rotavg3" not in grouped:
            raise RuntimeError(f"No pet_spice_rotavg3 windows found in {target_dir / 'outputs'}")
        windows = grouped["pet_spice_rotavg3"]
        shifts = sorted(window.shift for window in windows)
        expected = list(range(first_shift, last_shift + 1))
        if shifts != expected:
            raise RuntimeError(
                f"{target_dir}: expected shifts {first_shift}..{last_shift}, "
                f"got {shifts[:3]} ... {shifts[-3:]}"
            )
        diff_gp_results = ema.run_diff_gp_from_windows(source_dir, windows)
        for path in ema.save_ui_outputs(target_dir, source_dir, MODEL_TAG, windows, diff_gp_results):
            print(path)
        for path in ema.save_gpr_outputs(target_dir, source_dir, MODEL_TAG, windows, diff_gp_results):
            print(path)


if __name__ == "__main__":
    main()
