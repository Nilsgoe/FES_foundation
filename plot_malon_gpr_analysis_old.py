from __future__ import annotations

import argparse

import extend_malon_analysis as ema


def run_case(output_dir, diff_gp_source_dir, model_tag, windows, refit: bool) -> None:
    results = ema.get_diff_gp_results(output_dir, diff_gp_source_dir, model_tag, windows, refit)
    generated = ema.save_gpr_outputs(output_dir, diff_gp_source_dir, model_tag, windows, results)
    for path in generated:
        if "_analysis." in str(path):
            print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GPR-Umbrella-Sampling-Analysis-style malonaldehyde plots."
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Retrain diff_GP. By default, cached umbrella_gpr_* CSV predictions are reused.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ema.plot_style()
    for system_name in ema.SYSTEMS:
        system_dir = ema.ROOT / system_name
        grouped = ema.gather_windows(system_dir)
        for model_tag, windows in sorted(grouped.items()):
            run_case(system_dir / "analysis", system_dir, model_tag, windows, args.refit)

        for model_tag, csv_path, raw_outputs_dir in ema.VIPER_ANALYSIS_INPUTS[system_name]:
            if not csv_path.exists():
                continue
            windows = ema.load_windows_from_analysis_csv(model_tag, csv_path, raw_outputs_dir)
            run_case(csv_path.parent, system_dir, model_tag, windows, args.refit)


if __name__ == "__main__":
    main()
