"""
feature_tracking.py
───────────────────
Step 1 – SIFT feature detection + FLANN matching
Step 2 – Essential-matrix estimation and pose recovery (R, t)
"""

import numpy as np
import cv2 as cv


# ──────────────────────────────────────────────────────────────────────────────
# Feature matching
# ──────────────────────────────────────────────────────────────────────────────

def track_features(img1, img2, n_features=1500):
    """
    Detect SIFT keypoints in both images and match them with FLANN + ratio test.

    Returns:
        pts1, pts2 – matched (N,2) float32 arrays in img1 / img2
        Returns (None, None) when fewer than 8 good matches are found.
    """
    sift = cv.SIFT_create(nfeatures=n_features)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, None

    des1, des2 = np.float32(des1), np.float32(des2)
    flann   = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:          # Lowe's ratio test
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    if len(pts1) < 8:
        return None, None

    return np.float32(pts1), np.float32(pts2)


# ──────────────────────────────────────────────────────────────────────────────
# Pose estimation
# ──────────────────────────────────────────────────────────────────────────────

def estimate_pose(pts1, pts2, K):
    """
    Compute the Essential matrix (RANSAC) and recover relative R, t.

    Returns:
        R      – (3,3) rotation matrix
        t      – (3,1) translation (unit scale)
        pts1m  – inlier points from img1
        pts2m  – inlier points from img2
        Returns (None, None, None, None) on failure.
    """
    E, mask = cv.findEssentialMat(pts2, pts1, K,
                                   cv.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None, None, None

    pts1m = pts1[mask.ravel() == 1]
    pts2m = pts2[mask.ravel() == 1]

    if len(pts1m) < 8:
        return None, None, None, None

    _, R, t, _ = cv.recoverPose(E, pts1m, pts2m, K)
    return R, t, pts1m, pts2m
