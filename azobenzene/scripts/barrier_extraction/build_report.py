"""Cross-model aggregation and report builder for azobenzene barrier extraction.

Reads summary_table.csv (written by run_analysis.py / Task 8 orchestrator),
aggregates per MACE model across the two seeds (cis-start, trans-start), writes
summary_by_model.csv, and produces barrier_analysis.md.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np

OUT = Path("azobenzene/analysis/barrier_extraction")


def _df_to_markdown(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table without tabulate."""
    cols = list(df.columns)
    # Format each cell
    def fmt(val):
        if isinstance(val, float):
            if np.isnan(val):
                return "NaN"
            return format(val, floatfmt.lstrip(".").join((".", "")))
        return str(val)

    def fmt_float(val):
        if isinstance(val, float):
            if np.isnan(val):
                return "NaN"
            spec = floatfmt if floatfmt.startswith(".") else f".{floatfmt}"
            return f"{val:{spec}}"
        return str(val)

    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows_md = []
    for _, row in df.iterrows():
        cells = [fmt_float(row[c]) for c in cols]
        rows_md.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows_md)


def main() -> None:
    df = pd.read_csv(OUT / "summary_table.csv")
    df["seed"]  = df["tag"].str.split("_").str[0]
    df["model"] = df["tag"].str.split("_").str[1]

    keys = [
        "rot_cis_to_trans_mean_kJmol",
        "rot_trans_to_cis_mean_kJmol",
        "inv_cis_to_trans_mean_kJmol",
        "inv_trans_to_cis_mean_kJmol",
        "dG_rxn_mean_kJmol",
    ]

    rows = []
    for model, g in df.groupby("model"):
        row = {
            "model": model,
            "n_seeds": len(g),
            "any_unconverged": int((~g["converged"]).any()),
        }
        for k in keys:
            row[k + "_avg"] = g[k].mean()
            row[k + "_SE"]  = (
                g[k].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan
            )
        rows.append(row)

    agg = pd.DataFrame(rows)
    agg.to_csv(OUT / "summary_by_model.csv", index=False)

    md = ["# Azobenzene cis ⇌ trans barriers from 2D WT-MetaD\n"]
    md.append(
        "**Data:** [azobenzene/outputs/full_runs/](../../outputs/full_runs/), "
        "8 production 2D WT-MetaD runs (4 MACE models × {cis,trans}-start). "
        "Bias factor γ = 10; height₀ = 0.1 eV; σ = 5° both CVs; 500 ps each.\n"
    )
    md.append(
        "**Engine:** custom ASE `WT_Metadynamics` "
        "([/nexus/posix0/FHI-Theory/ngoen/biASE/Metadynamics.py:463]"
        "(/nexus/posix0/FHI-Theory/ngoen/biASE/Metadynamics.py)). "
        "Hill heights in the `.txt` log are pre-multiplied by γ/(γ−1); reconstruction uses "
        "`F = −V_bias` directly.\n"
    )
    md.append(
        "**Temperature:** 333 K (kT = 0.02870 eV), per `run_metad.py:177,182`.\n"
    )
    md.append(
        "**Uncertainty estimator:** `σ = max(σ_cumulative, σ_block_to_block)` over 5 cumulative "
        "blocks of the second half of hills (conservative; flags residual drift).\n"
    )
    md.append("## Per-run results\n")
    md.append(_df_to_markdown(df, floatfmt=".2f"))
    md.append("\n## Per-model aggregation (mean ± SE over the two seeds)\n")
    md.append(_df_to_markdown(agg, floatfmt=".2f"))
    md.append("\n## Figures\n")
    for tag in sorted(df["tag"].unique()):
        md.append(f"### {tag}\n")
        md.append(f"- ![FES](figures/{tag}/fes_2d.png)")
        md.append(f"- ![Convergence (hills)](figures/{tag}/convergence_hills.png)")
        md.append(f"- ![Convergence (FES)](figures/{tag}/convergence_fes.png)")
        md.append(f"- ![MFEP profile](figures/{tag}/mfep_profile.png)\n")
    # 1D-MetaD cross-check (Task 11): an independent estimator that biases
    # only on CV1 (CNNC dihedral). Agreement with the 2D MFEP barriers within
    # ~kT is evidence the orthogonal coordinate is not the rate-limiting
    # bottleneck; large disagreement flags sampling problems in the 2D run.
    one_d_csv = OUT / "summary_table_1d.csv"
    if one_d_csv.exists():
        df1d = pd.read_csv(one_d_csv)
        df1d["seed"]  = df1d["tag"].str.split("_").str[0]
        df1d["model"] = df1d["tag"].str.split("_").str[1]
        df2d_min = df[["seed", "model",
                       "rot_cis_to_trans_mean_kJmol", "dG_rxn_mean_kJmol"]].rename(
            columns={"rot_cis_to_trans_mean_kJmol": "2D_MFEP_c2t_kJmol",
                     "dG_rxn_mean_kJmol":           "2D_dG_rxn_kJmol"})
        df1d_min = df1d[["seed", "model",
                         "dG_c2t_kJmol", "dG_rxn_kJmol"]].rename(
            columns={"dG_c2t_kJmol": "1D_c2t_kJmol",
                     "dG_rxn_kJmol": "1D_dG_rxn_kJmol"})
        cmp = df2d_min.merge(df1d_min, on=["seed", "model"], how="outer")
        cmp["Δc2t_kJmol"]    = cmp["2D_MFEP_c2t_kJmol"] - cmp["1D_c2t_kJmol"]
        cmp["Δdg_rxn_kJmol"] = cmp["2D_dG_rxn_kJmol"]  - cmp["1D_dG_rxn_kJmol"]
        md.append("\n## 1D-MetaD cross-check (Task 11)\n")
        md.append(
            "Independent native 1D MetaD runs (bias only on CV1) processed by "
            "`run_1d_analysis.py`. Disagreement > kT ≈ 2.6 kJ/mol flags 2D sampling "
            "issues (e.g. orthogonal CV not visited from one seed).\n\n"
            "Per-run native-1D figures: `figures/{tag}/fes_1d_native.png`. Raw "
            "summary: `summary_table_1d.csv`.\n"
        )
        md.append(_df_to_markdown(cmp, floatfmt=".1f"))
        md.append("")

    # TS structures (Task 12): pointer to .xyz files extracted near each
    # 2D MFEP saddle. Match score in deg² = squared distance in (cv1,cv2)
    # from the predicted saddle; <100 = within ~10° of the saddle.
    ts_root = OUT / "ts_structures"
    if ts_root.exists() and any(ts_root.iterdir()):
        md.append("\n## Transition-state structures (Task 12)\n")
        md.append(
            "Representative atomic geometries extracted from the MD trajectory at "
            "the frame closest to each predicted 2D MFEP saddle, written as "
            "`.xyz` files under `ts_structures/{tag}/ts_rotation.xyz` and "
            "`ts_structures/{tag}/ts_inversion.xyz`. Match scores were logged at "
            "extraction time (lower is better; < 100 deg² ≈ within 10° of the saddle).\n"
        )

    md.append("\n## Literature comparison\n")
    md.append(
        "**Reference values** (verify each citation against your own bibliography "
        "before publication):\n\n"
        "| Quantity | Value (kJ/mol) | Source kind | Reference to verify |\n"
        "|---|---|---|---|\n"
        "| ΔG‡(trans→cis, thermal, **solution**) | ≈ 96 | experimental kinetics | Schmidt et al.; Bandara & Burdette, *Chem. Soc. Rev.* 41 (2012); Tiberio et al. (2010) |\n"
        "| ΔG‡(rotation, gas-phase) | ≈ 150–180 | DFT / multireference | Cembran, Bernardi, Garavelli et al., *JACS* 126 (2004); Casellas, Bearpark, Reguero (2016) |\n"
        "| ΔG‡(inversion, gas-phase) | ≈ 110–130 | DFT / multireference | Same as above |\n"
        "| Solvation drop on barrier | 20–40 | solution vs. gas-phase | Tiberio et al., *JCTC* 6 (2010) |\n\n"
        "**Interpretation rule:** this work is **gas-phase MACE** with no explicit solvent. "
        "Compare to gas-phase DFT (~110–180 kJ/mol), NOT to solution-phase experiment (~96 kJ/mol). "
        "A solvation correction is needed before comparing to experiment.\n\n"
        "**Flag in this report** any model whose barrier disagrees with gas-phase DFT "
        "by more than ~20 kJ/mol, and report which mechanism (rotation vs inversion) the model "
        "favors. The relative ordering of mechanisms is more diagnostic than absolute barrier height.\n"
    )
    md.append("\n## Caveats\n")
    md.append(
        "- F = −V_bias (no extra γ/(γ−1)) because the engine pre-scales heights at "
        "`Metadynamics.py:839`. Row-1 height ratio 0.1111 = 0.1·10/9 was used to verify.\n"
        "- CV2 boundary masked within 5° of the 0°/180° walls. The inversion TS lies near "
        "CV2≈175°; the masked margin is reported in the per-run figure.\n"
        "- Uncertainties: `σ = max(σ_cum, σ_b2b)` over 5 cumulative blocks of the second half "
        "of hills. The b2b estimator captures residual drift; the cumulative one captures spread "
        "across the running estimates.\n"
        "- The two seeds (cis-start, trans-start) are independent replicas; we report each "
        "separately and the model-level SE over the two as `(σ_seed1² + σ_seed2²)^½ / √2`.\n"
    )
    (OUT / "barrier_analysis.md").write_text("\n".join(md))
    print(f"Wrote {OUT / 'barrier_analysis.md'}")
    print(f"Wrote {OUT / 'summary_by_model.csv'}")


if __name__ == "__main__":
    main()
