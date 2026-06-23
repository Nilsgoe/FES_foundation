from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class MetadRun:
    path: Path
    tag: str                  # e.g. "cis_omol_2d"
    height0_eV: float         # nominal Height from header
    pace_steps: int
    step_size_fs: float
    bias_factor: float
    wt: bool
    time_fs: np.ndarray       # (n_hills,)
    cv1_deg: np.ndarray       # (n_hills,) CNNC dihedral
    cv2_deg: np.ndarray       # (n_hills,) NNC angle
    sigma1_deg: np.ndarray    # (n_hills,)
    sigma2_deg: np.ndarray
    height_eV: np.ndarray     # (n_hills,) ALREADY γ/(γ-1)-scaled
    reg_factor: np.ndarray    # (n_hills,) = h_i / (Height0 * γ/(γ-1))


_HEADER_KV = re.compile(r"(\w+)\s*=\s*(True|False|[-\d.eE+]+)")


def parse_bias_log(path: str | Path) -> MetadRun:
    path = Path(path)
    with path.open() as f:
        header = f.readline()
    if not header.startswith("!#"):
        raise ValueError(f"{path}: missing '!#' header line")
    kv = dict(_HEADER_KV.findall(header))
    data = np.loadtxt(path, skiprows=2)
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError(f"{path}: expected >=8 columns, got shape {data.shape}")

    # Restart-discontinuity guard
    dt = np.diff(data[:, 0])
    if (dt <= 0).any():
        drops = np.where(dt <= 0)[0]
        raise ValueError(
            f"{path}: time column non-monotonic at row(s) {drops[:5].tolist()}. "
            "Likely a restarted/concatenated log; deduplicate before reconstruction."
        )

    tag = _tag_from_filename(path)
    try:
        height0_eV   = float(kv["Height"])
        pace_steps   = int(kv["Pace"])
        step_size_fs = float(kv["Step_size"])
        bias_factor  = float(kv["Bias_factor"])
        wt           = (kv["WT"] == "True")
    except KeyError as exc:
        raise ValueError(f"{path}: missing header field {exc}") from exc
    return MetadRun(
        path=path,
        tag=tag,
        height0_eV=height0_eV,
        pace_steps=pace_steps,
        step_size_fs=step_size_fs,
        bias_factor=bias_factor,
        wt=wt,
        time_fs=data[:, 0],
        cv1_deg=data[:, 1],
        cv2_deg=data[:, 2],
        sigma1_deg=data[:, 3],
        sigma2_deg=data[:, 4],
        height_eV=data[:, 5],
        reg_factor=data[:, 7],
    )


def parse_bias_log_1d(path: str | Path) -> MetadRun:
    """Parse a 1D MetaD bias log (6-column: time CV1 sigma1 height biasfactor regfactor).

    cv2_deg and sigma2_deg are filled with zeros so the returned MetadRun is
    compatible with the dataclass but those fields must not be used.
    """
    path = Path(path)
    with path.open() as f:
        header = f.readline()
    if not header.startswith("#!") and not header.startswith("!#"):
        raise ValueError(f"{path}: missing header line")
    kv = dict(_HEADER_KV.findall(header))
    data = np.loadtxt(path, skiprows=2)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(f"{path}: expected >=6 columns for 1D log, got shape {data.shape}")

    dt = np.diff(data[:, 0])
    if (dt <= 0).any():
        drops = np.where(dt <= 0)[0]
        raise ValueError(
            f"{path}: time column non-monotonic at row(s) {drops[:5].tolist()}. "
            "Likely a restarted/concatenated log; deduplicate before reconstruction."
        )

    tag = _tag_from_filename(path)
    try:
        height0_eV   = float(kv["Height"])
        pace_steps   = int(kv["Pace"])
        step_size_fs = float(kv["Step_size"])
        bias_factor  = float(kv["Bias_factor"])
        wt           = (kv["WT"] == "True")
    except KeyError as exc:
        raise ValueError(f"{path}: missing header field {exc}") from exc
    n = data.shape[0]
    return MetadRun(
        path=path,
        tag=tag,
        height0_eV=height0_eV,
        pace_steps=pace_steps,
        step_size_fs=step_size_fs,
        bias_factor=bias_factor,
        wt=wt,
        time_fs=data[:, 0],
        cv1_deg=data[:, 1],
        cv2_deg=np.zeros(n),
        sigma1_deg=data[:, 2],
        sigma2_deg=np.zeros(n),
        height_eV=data[:, 3],
        reg_factor=data[:, 5],
    )


def _tag_from_filename(p: Path) -> str:
    # metad_azob_{cis|trans}_{model}_{1d|2d}_*.txt -> "{cis|trans}_{model}_{1|2}d"
    m = re.search(r"metad_azob_(cis|trans)_(\w+?)_(1d|2d)_", p.name)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"

    # Viper model logs use e.g. pet_spice_azob_cis_2d.bias or
    # sol3r_azob_trans_2d.bias.
    m = re.search(r"(\w+?)_azob_(cis|trans)_(1d|2d)", p.name)
    if m:
        return f"{m.group(2)}_{m.group(1)}_{m.group(3)}"

    raise ValueError(f"unrecognized filename pattern: {p.name}")
