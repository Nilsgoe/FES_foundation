# Metadynamics periodicity audit

Audit of angle-type CVs across the subdirectories of
`/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/` against the
bias engine in `/home/ngoen/biASE/Metadynamics.py` (`WT_Metadynamics`).

Date: 2026-04-21

## How the bias engine handles periodicity

`WT_Metadynamics.__init__` takes

- `wrapping`: list of booleans, one per CV
- `bounds`: tuple of `(low, high)` pairs, one per CV

If any entry of `wrapping` is True, the 1D/2D Gaussian dispatcher
(`gaussian_1D` / `gaussian_2D`) routes into the wrapped kernel
(`gaussian_1D_wrapped` / `gaussian_2D_wrapped`). In the 2D wrapped
kernel each axis is branched independently, so mixed
periodic/non-periodic CVs are supported.

The wrapped kernel uses the minimum-image displacement

```
diffs = (x - mean + 0.5 * P) % P - 0.5 * P   with   P = high - low
```

This is the standard PLUMED-style minimum-image for a periodic CV and
is correct when the CV is genuinely periodic with period `P`.

Important consequences:

1. Setting `wrapping=True` for a CV that is **not** periodic folds
   Gaussians that cross the interval boundary back into the interval,
   i.e. the bias deposited near the upper bound "leaks" to the lower
   bound and vice versa. Physically wrong for e.g. bond angles.
2. Setting `wrapping=False` for a CV that **is** periodic (e.g. a
   dihedral in `[-180, 180]`) deposits Gaussians that never wrap.
   Conformations near `+180` and `-180` are only bridged by Gaussian
   tails, biased CV values near the boundary accumulate two uncoupled
   histories, and free-energy reconstruction along the CV will show a
   spurious discontinuity at the seam.
3. The plain (non-WT) `Metadynamics` class has no wrapping at all.
   None of the runs below use it, but this is still a latent trap.

## Per-run findings

### OK — correct periodicity handling

- `azobenzene/run_metad.py`
  - 1D mode: CV = CNNC dihedral, `wrapping=[True]`,
    `bounds=((-180, 180),)`. Correct.
  - 2D mode: CV = (dihedral, CNN bond angle), `wrapping=[True, False]`,
    `bounds=((-180, 180), (0, 180))`. Correct — dihedral wraps with
    period 360, bond angle is treated as non-periodic.

- `trialanine_mace_metad/scripts/run_mace_trialanine.py`
  - CV = (φ, ψ) Ramachandran dihedrals, `wrapping=[True, True]`,
    `bounds=((-180, 180), (-180, 180))`. Correct.

- `malonaldehyd/run_MetaD.py`, `f-malonaldehyd/run_MetaD.py`
  - CV = `d(O–H) − d(O–H)` distance difference. Not angular, no
    periodicity. `wrapping` is not passed and defaults to `[False]`.
    Correct.

### BUG — periodicity mishandled

#### B1. `pp-azob/run_MetaD.py` (1D cis-dihedral test)

CV is a CNNC dihedral (returned in degrees, range `[-180, 180)`), but
the call site

```python
dyn = WT_Metadynamics(... cvs=compute_cv, std_dev=5, bias_height=.1,
                     interval_size=100, ..., well_temp=True,
                     bias_factor=10)
```

does not pass `wrapping` or `bounds`, so `wrapping` defaults to
`[False]` and the non-wrapped Gaussian kernel is used. A dihedral is
periodic with period 360°, so this run:

- never couples the `-180°` and `+180°` regions through the bias,
- will show a spurious barrier at the `±180°` seam in the
  reconstructed free energy,
- can accumulate a larger-than-needed bias near the seam because CV
  trajectories that cross it deposit disjoint histories on each side.

This is the same molecule/CV as the corresponding azobenzene run in
`azobenzene/run_metad.py`, where `wrapping=[True]` is set correctly,
which confirms the missing argument here is an oversight.

#### B2. `pp-azob/run_MetaD_2D_solv.py` and `pp-azob/run_MetaD_2D_solv_1.py` (2D, solvated)

Both files use

```python
cvs=compute_dihedral_and_angle,
std_dev=[5, 5],
...,
wrapping=[True, True],
bounds=((-180, 180), (0, 180)),
```

CV1 is the CNNC dihedral → correct with `wrapping[0]=True` and period
`360`.

CV2 is the CNN **bond angle** computed by `arccos`. Its value lives in
`[0°, 180°]`, and unlike a dihedral it is **not periodic**: a CNN
angle of `1°` and `179°` describe completely different geometries.
With `wrapping[1]=True` and `bounds=(0, 180)` the engine uses period
`P = 180` and applies

```
dy = (y - mean_y + 90) % 180 - 90
```

which folds the Gaussians across the interval. Consequences:

- A Gaussian deposited near `y = 5°` biases `y = 175°` just as
  strongly as `y = 5°`, which is unphysical.
- Near-linear geometries (`y ≈ 180°`) and near-collinear-flipped
  geometries (`y ≈ 0°`) are biased together, collapsing distinct
  regions of configuration space.
- Well-tempered reweighting / free-energy reconstruction along CV2
  will be wrong.

The CV-mode twin of this run in `azobenzene/run_metad.py` sets
`wrapping=[True, False]`, which is the correct handling and should be
mirrored here.

#### Suspected downstream effects

- Any FES computed from `metad_pp_azob_trans_2D_solv*.txt` along the
  CNN angle is quantitatively unreliable and qualitatively misleading.
- Restart via `input_file=metad_pp_azob_trans_2D_solv*.txt` inherits
  the flawed `wrapping` setting (the loader simply re-ingests the
  CV/reg-factor time series), so continuation runs perpetuate the bug.

### Other notes (not bugs but worth tracking)

- **Kernel cutoff / tail handling**: the wrapped kernel does not apply
  a finite Gaussian cutoff; it always uses the minimum image only.
  For `std_dev = 5°` and `P = 360°` the next-image contribution is
  `exp(-0.5 · (355/5)^2) ≈ 0`, so this is fine. For very wide
  Gaussians relative to the period (not the case here) one should sum
  images.
- **Force continuity at the seam**: `diffs^2` is continuous at the
  seam but its derivative jumps by `±P` there, producing a (very
  short-lived) force kick when the CV crosses the boundary. Same
  behaviour as PLUMED's periodic MetaD; acceptable.
- **Plain `Metadynamics` class**: no wrapping support. If anyone reuses
  it for angular CVs it will silently be wrong. Currently unused in
  these subdirs.
- **`bounds` semantics for a dihedral**: `(-180, 180)` matches the
  range of `arctan2` in degrees; `(0, 360)` would also work as long as
  the CV is shifted to the same range. The current choice is
  self-consistent.

## Fix plan

Objective: make every MetaD run in this tree declare periodicity
consistent with the physical CV.

### Step 1 — patch `pp-azob/run_MetaD.py` (1D dihedral)

Add the missing periodicity arguments to mirror `azobenzene/run_metad.py`:

```python
dyn = WT_Metadynamics(
    atoms,
    timestep=timestep,
    temperature_K=333,
    friction=.1,
    trajectory='metad_azob_cis_test.traj',
    fixcm=False,
    cvs=compute_cv,
    std_dev=5,
    bias_height=.1,
    interval_size=100,
    output_file='metad_azob_cis_test.txt',
    well_temp=True,
    bias_factor=10,
    wrapping=[True],
    bounds=((-180.0, 180.0),),
    max_bias=int(1e6),
)
```

### Step 2 — patch `pp-azob/run_MetaD_2D_solv.py` and `pp-azob/run_MetaD_2D_solv_1.py`

Change only the periodicity declaration; the CV function and all
other settings stay the same:

```python
wrapping=[True, False],
bounds=((-180.0, 180.0), (0.0, 180.0)),
```

The `bounds` tuple itself does not need to change — setting
`wrapping[1]=False` is enough to disable wrapping on CV2 regardless
of what `bounds[1]` says. Leaving `bounds[1]=(0,180)` is harmless and
keeps the declared physical range of the angle for documentation.

### Step 3 — flag previously produced solvated data

The existing `metad_pp_azob_trans_2D_solv*.txt`,
`metad_pp_azob_trans_2D_solv_cont*.txt`,
`metad_pp_azob_trans_2D_solv_1.txt`, and the corresponding `.traj`
files were produced with the incorrect angular wrapping.

Recommended action:

1. Move them to an `archive_wrapping_bug/` subfolder (do **not** feed
   them back as `input_file=` for the fixed runs).
2. Re-launch the 2D solvated MetaD from a clean equilibrated
   structure (`opt_solvated_pp-azob_trans_dmso_-25.xyz` or the latest
   `md_cu_eq_solvated_pp-azob_trans_dmso*.traj` frame) with the
   patched script.

### Step 4 — add a guard in `Metadynamics.py` (optional, recommended)

To prevent silent repeats, add a sanity check in
`WT_Metadynamics.__init__`:

```python
if self.wrapping is not None and self.bounds is not None:
    for i, w in enumerate(self.wrapping):
        if w:
            low, high = self.bounds[i]
            if not (high > low):
                raise ValueError(
                    f"wrapping[{i}]=True requires bounds[{i}]=(low, high) with high>low"
                )
```

and emit a warning when a user sets `wrapping=True` for a CV whose
declared bounds cover a range smaller than the Gaussian support:

```python
if w and (high - low) < 4 * self.std_dev[i]:
    warnings.warn(
        f"CV {i}: period {high-low} is small vs std_dev {self.std_dev[i]}."
        " Consider summing periodic images."
    )
```

Also consider adding a docstring note that bond angles are not
periodic and should keep `wrapping=False` even though their natural
range is `[0, 180]`.

### Step 5 — sanity check on restarts

Any restart (`input_file=...`) must pass the same `wrapping` /
`bounds` as the original run. Today the engine does not persist this
metadata in the text header — it only stores `Height`, `Pace`,
`Step_size`, `CV_Number`, `Bias_factor`, `WT`, `Stretch`. Optional
improvement: write `Wrapping=...` and `Bounds=...` into the header
and validate them on load. This would have caught the bug above
automatically on the continuation runs.

### Step 6 — regression test

Small, cheap test (1–2 ps) per fixed script:

- Confirm `metad_*.txt` CV column stays within the declared bounds.
- Visually check the deposited Gaussian trace does not show a
  discontinuity at the seam for dihedrals, and does not fold across
  for the bond angle in the pp-azob 2D run.

## Summary

- `pp-azob/run_MetaD.py`: dihedral CV run without `wrapping`. Fix:
  pass `wrapping=[True]`, `bounds=((-180, 180),)`.
- `pp-azob/run_MetaD_2D_solv.py` and
  `pp-azob/run_MetaD_2D_solv_1.py`: bond angle treated as periodic.
  Fix: `wrapping=[True, False]`.
- All other subdirs (azobenzene, trialanine, malonaldehyde,
  f-malonaldehyde) handle periodicity correctly.
- Engine itself is correct; the bugs are at the call sites.
- Previously generated pp-azob 2D solvated MetaD data is affected and
  should be rerun with the fix before being used for free-energy
  analysis.
