"""
geometry.py
───────────
3-D geometry helpers used inside the VO main loop:
  • triangulate       – project matched points into 3-D
  • hom_to_3d         – homogeneous → Cartesian + depth filter
  • relative_scale    – median-based scale recovery
  • reprojection_error – mean pixel error for a cloud against pose
"""

import numpy as np
import cv2 as cv


def triangulate(R, t, pt1, pt2, K):
    """
    Triangulate point correspondences from two camera poses.

    Args:
        R, t  – relative rotation (3,3) and translation (3,) for the second camera
        pt1   – (N,2) points in first  frame
        pt2   – (N,2) points in second frame
        K     – (3,3) intrinsic matrix

    Returns:
        cloud4 – (4,N) homogeneous 3-D points
    """
    Pr0 = K @ np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0]], dtype=np.float64)
    Pr1 = K @ np.hstack((R, t.reshape(3, 1)))
    return cv.triangulatePoints(Pr0, Pr1, pt1.T, pt2.T)  # (4, N)


def hom_to_3d(cloud4):
    """
    Convert homogeneous (4,N) cloud to Cartesian (3,N) and build a depth mask.

    Returns:
        xyz  – (3,N) Cartesian coordinates
        mask – boolean array, True for points with 0.1 < Z < 200
    """
    w   = cloud4[3, :] + 1e-8
    xyz = cloud4[:3, :] / w
    mask = (xyz[2, :] > 0.1) & (xyz[2, :] < 200)
    return xyz, mask


def relative_scale(c_old, c_new):
    """
    Estimate the metric scale between two successive point clouds using the
    median ratio of inter-point distances.

    Args:
        c_old, c_new – (3, N) Cartesian point clouds

    Returns:
        scale – positive float
    """
    s   = min(c_old.shape[1], c_new.shape[1])
    o, n = c_old[:, :s], c_new[:, :s]
    sc  = (np.linalg.norm(o - np.roll(o, 1, 1), axis=0) /
           (np.linalg.norm(n - np.roll(n, 1, 1), axis=0) + 1e-8))
    return float(np.median(sc))


def reprojection_error(pts2, cloud4, R_abs, t_abs, K):
    """
    Mean reprojection error of triangulated points against a given absolute pose.

    Args:
        pts2   – (N,2) observed image points in the current frame
        cloud4 – (4,N) homogeneous 3-D points
        R_abs  – (3,3) absolute rotation
        t_abs  – (3,)  absolute translation
        K      – (3,3) intrinsic matrix

    Returns:
        mean pixel error (float); inf if too few valid points
    """
    xyz, mask = hom_to_3d(cloud4)
    if mask.sum() < 4:
        return float('inf')

    pts2_ok = pts2.reshape(-1, 2)[mask]        # (M, 2)
    xyz_ok  = xyz[:, mask]                      # (3, M)

    P   = np.hstack((R_abs, t_abs.reshape(3, 1)))
    Xh  = np.vstack((xyz_ok, np.ones((1, xyz_ok.shape[1]))))
    proj = K @ (P @ Xh)
    proj = (proj[:2, :] / (proj[2:3, :] + 1e-8)).T   # (M, 2)

    return float(np.mean(np.linalg.norm(proj - pts2_ok, axis=1)))
