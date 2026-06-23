from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azobenzene.scripts.barrier_extraction.basins_mfep import enumerate_pathways
from azobenzene.scripts.barrier_extraction.fes_io import parse_bias_log
from azobenzene.scripts.barrier_extraction.fes_reconstruct import default_grid, reconstruct_fes_2d
from azobenzene.scripts.barrier_extraction.uncertainty import _pick_basin


ANALYSIS_TIME_NS = float(os.environ.get("AZOB_ANALYSIS_TIME_NS", "2.0"))
TIME_DIR_LABEL = "2ns" if abs(ANALYSIS_TIME_NS - 2.0) < 1e-9 else f"{ANALYSIS_TIME_NS:g}ns"
TIME_FILE_LABEL = f"{ANALYSIS_TIME_NS:.1f}ns"

OUT_DIR = ROOT / "azobenzene" / "analysis" / "model_comparison" / f"{TIME_DIR_LABEL}_reconstructed"
PLOT_DIR = ROOT / "azobenzene" / "analysis" / "model_comparison" / f"{TIME_DIR_LABEL}_8model_paper"
PATH_DIR = OUT_DIR / "paths"

MAX_TIME_FS = ANALYSIS_TIME_NS * 1_000_000.0
GRID_DEG = 1.0
PLOT_LEVELS = 13
EV_TO_KJMOL = 96.485
EV_TO_KCALMOL = 23.0609

SYSTEMS = ("cis", "trans")
MODELS = ("off", "mh1", "polar", "pet_spice", "so3lr", "scratch", "ft_so3lr", "ft_mh1")
LABELS = {
    "off": "MACE-OFF",
    "mh1": "MACE-MH1",
    "polar": "MACE-Polar",
    "pet_spice": "PET-SPICE",
    "so3lr": "SO3LR",
    "scratch": "MACE from scratch",
    "ft_so3lr": "Fine-tuned SO3LR",
    "ft_mh1": "Fine-tuned MACE-MH1",
}


@dataclass(frozen=True)
class Record:
    system: str
    model: str
    site: str
    path: Path
    fallback_npz: Path | None = None


def _pet_spice_candidates(system: str) -> tuple[Path, ...]:
    return (
        Path("/work/gpuviper_ptmp/Enhanced_sampling/upet/azobenzene")
        / f"pet_spice_{system}_2d"
        / "outputs"
        / f"pet_spice_azob_{system}_2d.bias",
        ROOT
        / "viper_analysis"
        / "upet"
        / "azobenzene"
        / f"pet_spice_{system}_2d"
        / f"pet_spice_azob_{system}_2d.bias",
    )


def _choose_longest_existing(paths: tuple[Path, ...]) -> Path:
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError("No existing candidate files:\n" + "\n".join(map(str, paths)))
    if len(existing) == 1:
        return existing[0]
    parsed = [(float(parse_bias_log(path).time_fs[-1]), path) for path in existing]
    return max(parsed, key=lambda item: item[0])[1]


def load_records() -> list[Record]:
    jobs = {"off": ("off", "33975"), "mh1": ("mh1", "33030"), "polar": ("polar", "33031")}
    records: list[Record] = []
    for system in SYSTEMS:
        task = "0" if system == "cis" else "1"
        for model, (filename_model, job) in jobs.items():
            records.append(
                Record(
                    system,
                    model,
                    "raccoon",
                    ROOT
                    / "azobenzene"
                    / "outputs"
                    / f"metad_azob_{system}_{filename_model}_2d_raccoon_{filename_model}_job{job}_task{task}_gpu1.txt",
                )
            )
        records.extend(
            [
                Record(
                    system,
                    "pet_spice",
                    "viper",
                    _choose_longest_existing(_pet_spice_candidates(system)),
                    ROOT
                    / "azobenzene"
                    / "analysis"
                    / "model_comparison"
                    / "2ns_7model_paper"
                    / f"azobenzene_{system}_pet_spice_2d.npz",
                ),
                Record(
                    system,
                    "so3lr",
                    "viper",
                    ROOT
                    / "viper_analysis"
                    / "sol3r"
                    / "azobenzene"
                    / f"{system}_2d"
                    / f"sol3r_azob_{system}_2d.bias",
                ),
                Record(
                    system,
                    "scratch",
                    "raccoon",
                    ROOT / "raccoon_mace_scratch_azob" / "outputs" / f"mace_scratch_azob_{system}_2d.bias",
                ),
                Record(
                    system,
                    "ft_so3lr",
                    "viper",
                    ROOT / "viper_ft_so3lr_azob" / "outputs" / f"ft_so3lr_azob_{system}_2d.bias",
                ),
                Record(
                    system,
                    "ft_mh1",
                    "raccoon",
                    ROOT
                    / "raccoon_mace_ft_mh1_azob"
                    / "outputs"
                    / f"mace_ft_mh1_azob_{system}_2d.bias",
                ),
            ]
        )
    missing = [record.path for record in records if not record.path.exists()]
    if missing:
        raise FileNotFoundError("Missing bias logs:\n" + "\n".join(map(str, missing)))
    return records


def _convert_columns(row: dict[str, str | float | int]) -> dict[str, str | float | int]:
    for key in (
        "F_cis",
        "F_trans",
        "dG_cis_minus_trans",
        "rotation_cis_to_trans",
        "rotation_trans_to_cis",
        "inversion_cis_to_trans",
        "inversion_trans_to_cis",
        "unconstrained_cis_to_trans",
        "unconstrained_trans_to_cis",
    ):
        value = float(row[f"{key}_eV"])
        row[f"{key}_kJmol"] = value * EV_TO_KJMOL
        row[f"{key}_kcalmol"] = value * EV_TO_KCALMOL
    return row


def _write_path_csv(system: str, model: str, name: str, cv1: np.ndarray, cv2: np.ndarray, F: np.ndarray, info: dict) -> None:
    n2 = F.shape[1]
    idx = info["path_idx"]
    ii = idx // n2
    jj = idx % n2
    out = PATH_DIR / f"azobenzene_{system}_{model}_2d_{TIME_FILE_LABEL}_{name}_path.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "cv1_deg", "cv2_deg", "F_eV", "F_kJmol", "F_kcalmol"))
        writer.writeheader()
        for step, (i, j, f_val) in enumerate(zip(ii, jj, info["F_path"], strict=True)):
            writer.writerow(
                {
                    "step": step,
                    "cv1_deg": cv1[i],
                    "cv2_deg": cv2[j],
                    "F_eV": float(f_val),
                    "F_kJmol": float(f_val) * EV_TO_KJMOL,
                    "F_kcalmol": float(f_val) * EV_TO_KCALMOL,
                }
            )


def _path_ts(cv1: np.ndarray, cv2: np.ndarray, F: np.ndarray, info: dict) -> dict[str, float]:
    n2 = F.shape[1]
    idx = info["path_idx"]
    ii = idx // n2
    jj = idx % n2
    local_max = int(np.argmax(info["F_path"]))
    return {
        "cv1_deg": float(cv1[ii[local_max]]),
        "cv2_deg": float(cv2[jj[local_max]]),
        "F_eV": float(info["F_path"][local_max]),
        "F_kJmol": float(info["F_path"][local_max]) * EV_TO_KJMOL,
        "F_kcalmol": float(info["F_path"][local_max]) * EV_TO_KCALMOL,
    }


def _load_reconstructed_fallback(record: Record) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, str] | None:
    if record.fallback_npz is None or not record.fallback_npz.exists():
        return None
    data = np.load(record.fallback_npz)
    actual_time_fs = float(data["actual_time_fs"])
    if actual_time_fs < MAX_TIME_FS:
        return None
    return (
        np.asarray(data["cv1_deg"], dtype=float),
        np.asarray(data["cv2_deg"], dtype=float),
        np.asarray(data["free_energy_eV"], dtype=float),
        int(data["n_hills"]),
        actual_time_fs,
        f"{record.path}; fallback_reconstructed_fes={record.fallback_npz}",
    )


def _load_surface(record: Record) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, str]:
    run = parse_bias_log(record.path)
    n_hills = min(len(run.time_fs), int(np.searchsorted(run.time_fs, MAX_TIME_FS, side="right")))
    if n_hills == 0:
        raise ValueError(f"No hills selected for {record.path}")
    actual_time_fs = float(run.time_fs[n_hills - 1])
    fallback = _load_reconstructed_fallback(record)
    if fallback is not None and actual_time_fs < MAX_TIME_FS:
        return fallback
    cv1, cv2 = default_grid(GRID_DEG, GRID_DEG)
    F = reconstruct_fes_2d(run, cv1, cv2, n_hills=n_hills)
    return cv1, cv2, F, n_hills, actual_time_fs, str(record.path)


def process_record(record: Record) -> tuple[dict[str, str | float | int], tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    cv1, cv2, F, n_hills, actual_time_fs, source_path = _load_surface(record)

    cis_min = _pick_basin(F, cv1, cv2, target=(0.0, 120.0))
    trans_min = _pick_basin(F, cv1, cv2, target=(180.0, 120.0))
    F_cis = float(F[cis_min])
    F_trans = float(F[trans_min])
    paths = enumerate_pathways(F, cv1, cv2, cis_min, trans_min)

    np.savez_compressed(
        OUT_DIR / f"azobenzene_{record.system}_{record.model}_2d_{TIME_FILE_LABEL}_reconstructed_fes.npz",
        cv1_deg=cv1,
        cv2_deg=cv2,
        free_energy_eV=F,
        cis_min_ij=np.array(cis_min),
        trans_min_ij=np.array(trans_min),
        n_hills=n_hills,
        actual_time_fs=actual_time_fs,
        source=source_path,
    )

    for path_name, path_info in paths.items():
        _write_path_csv(record.system, record.model, path_name, cv1, cv2, F, path_info)

    rot_ts = _path_ts(cv1, cv2, F, paths["rotation"])
    inv_ts = _path_ts(cv1, cv2, F, paths["inversion"])
    row: dict[str, str | float | int] = {
        "system": record.system,
        "model": record.model,
        "label": LABELS[record.model],
        "source": record.site,
        "bias_log": source_path,
        "n_hills": n_hills,
        "last_time_fs": actual_time_fs,
        "last_time_ns": actual_time_fs / 1e6,
        "cis_min_cv1_deg": float(cv1[cis_min[0]]),
        "cis_min_cv2_deg": float(cv2[cis_min[1]]),
        "trans_min_cv1_deg": float(cv1[trans_min[0]]),
        "trans_min_cv2_deg": float(cv2[trans_min[1]]),
        "F_cis_eV": F_cis,
        "F_trans_eV": F_trans,
        "dG_cis_minus_trans_eV": F_cis - F_trans,
        "rotation_cis_to_trans_eV": float(paths["rotation"]["barrier_eV"]),
        "rotation_trans_to_cis_eV": float(np.max(paths["rotation"]["F_path"]) - F_trans),
        "inversion_cis_to_trans_eV": float(paths["inversion"]["barrier_eV"]),
        "inversion_trans_to_cis_eV": float(np.max(paths["inversion"]["F_path"]) - F_trans),
        "unconstrained_cis_to_trans_eV": float(paths["unconstrained"]["barrier_eV"]),
        "unconstrained_trans_to_cis_eV": float(np.max(paths["unconstrained"]["F_path"]) - F_trans),
        "rotation_ts_cv1_deg": rot_ts["cv1_deg"],
        "rotation_ts_cv2_deg": rot_ts["cv2_deg"],
        "rotation_ts_F_eV": rot_ts["F_eV"],
        "rotation_ts_F_kJmol": rot_ts["F_kJmol"],
        "rotation_ts_F_kcalmol": rot_ts["F_kcalmol"],
        "inversion_ts_cv1_deg": inv_ts["cv1_deg"],
        "inversion_ts_cv2_deg": inv_ts["cv2_deg"],
        "inversion_ts_F_eV": inv_ts["F_eV"],
        "inversion_ts_F_kJmol": inv_ts["F_kJmol"],
        "inversion_ts_F_kcalmol": inv_ts["F_kcalmol"],
    }
    return _convert_columns(row), (cv1, cv2, F), {"cis": cis_min, "trans": trans_min, **paths}


def _path_segments(cv1: np.ndarray, cv2: np.ndarray, F: np.ndarray, info: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    n2 = F.shape[1]
    idx = info["path_idx"]
    ii = idx // n2
    jj = idx % n2
    x = cv1[ii].astype(float)
    y = cv2[jj].astype(float)
    jumps = np.where(np.abs(np.diff(x)) > 180.0)[0] + 1
    cuts = np.concatenate(([0], jumps, [x.size]))
    return [(x[a:b], y[a:b]) for a, b in zip(cuts[:-1], cuts[1:], strict=True) if b - a > 1]


def make_path_figure(
    system: str,
    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    path_data: dict[tuple[str, str], dict],
    vmax: float,
    with_paths: bool,
) -> None:
    fig = plt.figure(figsize=(15.0, 8.2), facecolor="white")
    panel_width = 0.185
    panel_height = 0.385
    top_y = 0.555
    bottom_y = 0.075
    x_positions = (0.035, 0.255, 0.475, 0.695)
    positions = tuple((x, top_y, panel_width, panel_height) for x in x_positions) + tuple(
        (x, bottom_y, panel_width, panel_height) for x in x_positions
    )
    contour = None
    levels = np.linspace(0.0, vmax, PLOT_LEVELS)

    for panel_index, (model, position) in enumerate(zip(MODELS, positions, strict=True)):
        ax = fig.add_axes(position)
        cv1, cv2, F = surfaces[(system, model)]
        pdata = path_data[(system, model)]
        actual_ns = float(pdata["last_time_ns"])
        contour = ax.contourf(cv1, cv2, F.T, levels=levels, cmap="RdBu_r")
        if with_paths:
            for x, y in _path_segments(cv1, cv2, F, pdata["unconstrained"]):
                ax.plot(x, y, color="white", lw=2.4, solid_capstyle="round", zorder=5)
                ax.plot(x, y, color="black", lw=1.2, solid_capstyle="round", zorder=6)
            for basin_name, marker in (("cis", "o"), ("trans", "s")):
                i, j = pdata[basin_name]
                ax.plot(cv1[i], cv2[j], marker, ms=5.5, mfc="white", mec="black", mew=0.9, zorder=7)
        ax.set_title(f"{LABELS[model]}\n{actual_ns:.2f} ns", pad=6)
        ax.set_xlim(-180, 180)
        ax.set_ylim(60, 180)
        ax.set_xticks((-180, -90, 0, 90, 180))
        ax.set_yticks((60, 90, 120, 150, 180))
        ax.set_xlabel(r"CNNC dihedral, $\phi$ (deg)")
        ax.set_ylabel(r"CNN angle, $\theta$ (deg)" if panel_index in (0, 4) else "")
        ax.tick_params(direction="in", top=True, right=True, length=4.5, width=1.0)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5, width=0.8)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    colorbar_ax = fig.add_axes((0.925, bottom_y, 0.018, top_y + panel_height - bottom_y))
    cbar = fig.colorbar(contour, cax=colorbar_ax)
    cbar.set_label("Relative free energy (eV)")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.outline.set_linewidth(1.0)
    cbar.ax.tick_params(direction="out", length=4, width=0.9)
    suffix = "_with_paths" if with_paths else ""
    for extension in ("png", "pdf"):
        fig.savefig(
            PLOT_DIR / f"azobenzene_{system}_2d_8model_paper{suffix}.{extension}",
            dpi=600,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def write_csv(rows: list[dict[str, str | float | int]]) -> None:
    out = OUT_DIR / f"azobenzene_2d_{TIME_FILE_LABEL}_barriers.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_stationary_points_csv(rows: list[dict[str, str | float | int]]) -> None:
    out = OUT_DIR / f"azobenzene_2d_{TIME_FILE_LABEL}_minima_transition_states.csv"
    fieldnames = (
        "system",
        "model",
        "label",
        "last_time_ns",
        "point",
        "cv1_CNNC_deg",
        "cv2_CNN_deg",
        "F_eV",
        "F_kJmol",
        "F_kcalmol",
        "source",
        "bias_log",
    )
    point_specs = (
        ("cis_minimum", "cis_min_cv1_deg", "cis_min_cv2_deg", "F_cis_eV", "F_cis_kJmol", "F_cis_kcalmol"),
        ("trans_minimum", "trans_min_cv1_deg", "trans_min_cv2_deg", "F_trans_eV", "F_trans_kJmol", "F_trans_kcalmol"),
        (
            "rotation_transition_state",
            "rotation_ts_cv1_deg",
            "rotation_ts_cv2_deg",
            "rotation_ts_F_eV",
            "rotation_ts_F_kJmol",
            "rotation_ts_F_kcalmol",
        ),
        (
            "inversion_transition_state",
            "inversion_ts_cv1_deg",
            "inversion_ts_cv2_deg",
            "inversion_ts_F_eV",
            "inversion_ts_F_kJmol",
            "inversion_ts_F_kcalmol",
        ),
    )
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for point, cv1_key, cv2_key, f_ev_key, f_kj_key, f_kcal_key in point_specs:
                writer.writerow(
                    {
                        "system": row["system"],
                        "model": row["model"],
                        "label": row["label"],
                        "last_time_ns": row["last_time_ns"],
                        "point": point,
                        "cv1_CNNC_deg": row[cv1_key],
                        "cv2_CNN_deg": row[cv2_key],
                        "F_eV": row[f_ev_key],
                        "F_kJmol": row[f_kj_key],
                        "F_kcalmol": row[f_kcal_key],
                        "source": row["source"],
                        "bias_log": row["bias_log"],
                    }
                )


def write_summary_and_latex(rows: list[dict[str, str | float | int]]) -> None:
    summary_rows = []
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        c2t = np.array([float(row["unconstrained_cis_to_trans_eV"]) for row in subset], dtype=float)
        t2c = np.array([float(row["unconstrained_trans_to_cis_eV"]) for row in subset], dtype=float)
        dg = np.array([float(row["dG_cis_minus_trans_eV"]) for row in subset], dtype=float)
        summary_rows.append(
            {
                "model": model,
                "label": LABELS[model],
                "n_replicates": len(subset),
                "cis_to_trans_barrier_eV_mean": float(c2t.mean()),
                "cis_to_trans_barrier_eV_std": float(c2t.std(ddof=1)) if c2t.size > 1 else 0.0,
                "trans_to_cis_barrier_eV_mean": float(t2c.mean()),
                "trans_to_cis_barrier_eV_std": float(t2c.std(ddof=1)) if t2c.size > 1 else 0.0,
                "deltaG_cis_minus_trans_eV_mean": float(dg.mean()),
                "deltaG_cis_minus_trans_eV_std": float(dg.std(ddof=1)) if dg.size > 1 else 0.0,
                "min_actual_time_ns": min(float(row["last_time_ns"]) for row in subset),
            }
        )

    with (OUT_DIR / f"azobenzene_2d_{TIME_FILE_LABEL}_barrier_histogram_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Model & $\Delta G_{\mathrm{cis-trans}}$ (eV) & $\Delta G^\ddagger_{\mathrm{c\to t}}$ (eV) \\",
        r"\hline",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['label']} & "
            f"{row['deltaG_cis_minus_trans_eV_mean']:.3f} $\\pm$ {row['deltaG_cis_minus_trans_eV_std']:.3f} & "
            f"{row['cis_to_trans_barrier_eV_mean']:.3f} $\\pm$ {row['cis_to_trans_barrier_eV_std']:.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    (OUT_DIR / f"azobenzene_2d_{TIME_FILE_LABEL}_barrier_latex_table.tex").write_text("\n".join(lines))


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 1.1,
            "axes.titleweight": "medium",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    PATH_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float | int]] = []
    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    path_data: dict[tuple[str, str], dict] = {}
    plot_values: list[np.ndarray] = []

    for record in load_records():
        row, surface, pdata = process_record(record)
        rows.append(row)
        surfaces[(record.system, record.model)] = surface
        pdata["last_time_ns"] = row["last_time_ns"]
        path_data[(record.system, record.model)] = pdata
        F = surface[2]
        plot_values.append(F[np.isfinite(F)].ravel())

    write_csv(rows)
    write_stationary_points_csv(rows)
    write_summary_and_latex(rows)
    vmax = float(np.ceil(np.percentile(np.concatenate(plot_values), 99.9) * 10.0) / 10.0)
    for system in SYSTEMS:
        make_path_figure(system, surfaces, path_data, vmax, with_paths=False)
        make_path_figure(system, surfaces, path_data, vmax, with_paths=True)

    print(f"Wrote {OUT_DIR / f'azobenzene_2d_{TIME_FILE_LABEL}_barriers.csv'}")
    print(f"Wrote {OUT_DIR / f'azobenzene_2d_{TIME_FILE_LABEL}_barrier_histogram_summary.csv'}")
    print(f"Wrote {OUT_DIR / f'azobenzene_2d_{TIME_FILE_LABEL}_minima_transition_states.csv'}")
    print(f"Wrote {OUT_DIR / f'azobenzene_2d_{TIME_FILE_LABEL}_barrier_latex_table.tex'}")
    print(f"Wrote path overlays in {PLOT_DIR}")


if __name__ == "__main__":
    main()
