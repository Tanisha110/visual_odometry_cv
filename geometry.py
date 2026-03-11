"""
geometry.py
───────────
3-D geometry helpers for the VO pipeline.

pose_update() uses the CORRECT accumulation order matching the C++ reference:
    t_f = t_f + scale * (R_f @ t)   ← translate in current world frame
    R_f = R_f @ R                    ← post-multiply (append new rotation)

NOTE: The rotation accumulation MUST be R_f @ R (post-multiply), NOT R @ R_f.
Post-multiply means "apply R in the current camera frame", which matches how
cv2.recoverPose returns R (rotation of new frame w.r.t. old frame).
"""

import numpy as np
import cv2 as cv


# ──────────────────────────────────────────────────────────────────────────────
# Pose accumulation  (the most critical piece)
# ──────────────────────────────────────────────────────────────────────────────

def pose_update(R_f, t_f, R, t, scale):
    """
    Accumulate relative pose (R, t) into the global pose (R_f, t_f).

    Matches C++ reference exactly:
        t_f  ←  t_f + scale * (R_f @ t)    # translate in world frame
        R_f  ←  R_f @ R                     # POST-multiply (append rotation)

    Args:
        R_f   – (3,3) accumulated global rotation
        t_f   – (3,)  accumulated global translation
        R     – (3,3) relative rotation  (current frame w.r.t. previous)
        t     – (3,)  relative translation (unit vector from recoverPose)
        scale – metric scale (float)

    Returns:
        R_f_new (3,3), t_f_new (3,)
    """
    t_f_new = t_f + scale * (R_f @ t.flatten())
    R_f_new = R_f @ R          # ← POST-multiply, NOT R @ R_f
    return R_f_new, t_f_new


def scale_is_valid(t, scale, min_scale=0.1):
    """
    Replicate the C++ guard:
        scale > 0.1
        AND |t_z| > |t_x|
        AND |t_z| > |t_y|
    i.e. the dominant motion is forward (along Z), not sideways.
    """
    tv = t.flatten()
    return (scale > min_scale
            and abs(tv[2]) > abs(tv[0])
            and abs(tv[2]) > abs(tv[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Triangulation helpers
# ──────────────────────────────────────────────────────────────────────────────

def triangulate(R, t, pt1, pt2, K):
    """
    Triangulate correspondences. Returns (4,N) homogeneous cloud.
    """
    Pr0 = K @ np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0]], dtype=np.float64)
    Pr1 = K @ np.hstack((R, t.reshape(3, 1)))
    return cv.triangulatePoints(Pr0, Pr1, pt1.T, pt2.T)


def hom_to_3d(cloud4):
    """
    Homogeneous (4,N) → Cartesian (3,N) with depth mask (0.1 < Z < 200).
    Returns xyz (3,N) and boolean mask (N,).
    """
    w    = cloud4[3, :] + 1e-8
    xyz  = cloud4[:3, :] / w
    mask = (xyz[2, :] > 0.1) & (xyz[2, :] < 200)
    return xyz, mask


def relative_scale(c_old, c_new):
    """
    Median ratio of inter-point distances between two successive clouds.
    c_old, c_new are (3,N) arrays from hom_to_3d.
    Returns 1.0 safely when there are too few points.
    """
    s = min(c_old.shape[1], c_new.shape[1])
    if s < 2:
        return 1.0
    o, n = c_old[:, :s], c_new[:, :s]
    denom = np.linalg.norm(n - np.roll(n, 1, 1), axis=0) + 1e-8
    sc    = np.linalg.norm(o - np.roll(o, 1, 1), axis=0) / denom
    return float(np.median(sc))