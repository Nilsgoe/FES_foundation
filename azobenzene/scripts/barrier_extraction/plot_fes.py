from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

EV_TO_KJMOL = 96.485
EV_TO_KCALMOL = 23.061


def plot_2d_fes(F_eV, cv1, cv2, out_png: Path,
                title: str,
                basins: dict | None = None,
                paths: dict | None = None,
                contour_kJ: float = 5.0,
                f_max_kJ: float = 250.0):
    F_kJ = F_eV * EV_TO_KJMOL
    levels = np.arange(0, f_max_kJ + contour_kJ, contour_kJ)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    pcm = ax.pcolormesh(cv1, cv2, F_kJ.T, cmap="viridis",
                        shading="auto", vmin=0, vmax=f_max_kJ)
    cs = ax.contour(cv1, cv2, F_kJ.T, levels=levels, colors="k",
                    linewidths=0.4, alpha=0.5)
    ax.clabel(cs, levels=levels[::4], fmt="%d", fontsize=7)
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label("F (kJ/mol)  |  multiply by 0.239 for kcal/mol")

    if basins:
        for name, (i, j) in basins.items():
            ax.plot(cv1[i], cv2[j], "wo", mec="k", ms=8)
            ax.annotate(name, (cv1[i], cv2[j]),
                        textcoords="offset points", xytext=(5, 5), color="white",
                        fontsize=9, fontweight="bold")
    if paths:
        n1, n2 = F_eV.shape
        styles = {"rotation": ("-", "tab:red"), "inversion": ("--", "tab:orange"),
                  "unconstrained": (":", "white")}
        for name, info in paths.items():
            idx = info["path_idx"]
            ii, jj = idx // n2, idx % n2
            ls, col = styles.get(name, ("-", "white"))
            ax.plot(cv1[ii], cv2[jj], ls, color=col, lw=1.5, label=name)
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("CV1: CNNC dihedral (°)")
    ax.set_ylabel("CV2: NNC angle (°)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_convergence_hills(time_fs, heights_eV, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(time_fs / 1000.0, heights_eV * EV_TO_KJMOL, lw=0.6)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("hill height (kJ/mol, log)")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)


def plot_convergence_fes(snapshots: dict, cv1, cv2, out_png: Path, title: str):
    # 1D slice at the mean CV2 of the minimum-energy path is a robust 1-line summary.
    fig, ax = plt.subplots(figsize=(7, 4))
    for frac, F in sorted(snapshots.items()):
        F_kJ = F * EV_TO_KJMOL
        # take min over CV2 -> shows the 2D-projected-down profile vs CV1
        ax.plot(cv1, F_kJ.min(axis=1), label=f"{int(frac*100)}%")
    ax.set_xlabel("CV1: CNNC dihedral (°)")
    ax.set_ylabel("min over CV2 of F (kJ/mol)")
    ax.set_title(title)
    ax.legend(title="hills used")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)


def plot_mfep_profile(paths: dict, F_eV, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, info in paths.items():
        Fp = info["F_path"] * EV_TO_KJMOL
        s = np.arange(Fp.size)
        ax.plot(s, Fp - Fp.min(), label=f"{name} (ΔG‡ = {info['barrier_eV']*EV_TO_KJMOL:.1f} kJ/mol)")
    ax.set_xlabel("path index")
    ax.set_ylabel("F along MFEP (kJ/mol)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
