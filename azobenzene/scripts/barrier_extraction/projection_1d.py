import numpy as np
from scipy.special import logsumexp


def project_to_cv1(F_2d: np.ndarray, cv1_grid, cv2_grid,
                   kT_eV: float = 0.02870,   # 333 K production temperature
                   cv2_mask=None) -> np.ndarray:
    """F_1D(cv1) = -kT ln sum_j exp(-F(cv1,cv2_j)/kT) * Δcv2  (+ const)

    cv2_mask: optional boolean array over cv2 grid to exclude wall-region.
    """
    d_cv2 = float(np.median(np.diff(cv2_grid)))
    F = F_2d.copy()
    if cv2_mask is not None:
        F[:, ~cv2_mask] = np.inf  # exclude masked region
    # logsumexp over CV2 axis
    F1 = -kT_eV * (logsumexp(-F / kT_eV, axis=1) + np.log(d_cv2))
    F1 -= F1.min()
    return F1


def cv2_wall_mask(cv2_grid, margin_deg: float = 5.0) -> np.ndarray:
    """True for the interior; False within `margin_deg` of either boundary."""
    return (cv2_grid >= cv2_grid.min() + margin_deg) & \
           (cv2_grid <= cv2_grid.max() - margin_deg)
