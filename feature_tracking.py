"""
feature_tracking.py
───────────────────
Step 1 – FAST feature detection + Lucas-Kanade optical flow tracking
Step 2 – Essential-matrix estimation and pose recovery (R, t)

Inspired by: Avi Singh's monocular VO (C++ reference implementation)
  - FAST is much faster than SIFT and good enough for tracking
  - LK optical flow tracks existing points frame-to-frame (no descriptor matching)
  - Redetection is triggered when tracked point count drops below MIN_FEATURES
"""

import numpy as np
import cv2 as cv

MIN_FEATURES = 2000   # re-detect when tracked points fall below this


def detect_features(img):
    """
    Detect FAST keypoints and return them as (N,2) float32 array.
    """
    kps = cv.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)\
             .detect(img, None)
    if not kps:
        return np.empty((0, 2), dtype=np.float32)
    pts = cv.KeyPoint_convert(kps)          # → (N,2) float32
    return pts


def track_features(img1, img2, pts1):
    """
    Track points from img1 into img2 using Lucas-Kanade optical flow.
    Automatically discards points that fail tracking or leave the frame.

    Args:
        img1, img2 – consecutive grayscale frames
        pts1       – (N,2) float32 points detected in img1

    Returns:
        pts1_good – (M,2) subset of pts1 that tracked successfully
        pts2_good – (M,2) corresponding positions in img2
        Returns (None, None) if fewer than 8 points survive.
    """
    if pts1 is None or len(pts1) == 0:
        return None, None

    pts1 = pts1.reshape(-1, 1, 2).astype(np.float32)

    lk_params = dict(
        winSize      = (21, 21),
        maxLevel     = 3,
        criteria     = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
        minEigThreshold = 1e-4,
    )
    pts2, status, _ = cv.calcOpticalFlowPyrLK(img1, img2, pts1, None, **lk_params)

    # keep only successfully tracked points that stayed inside the frame
    h, w = img2.shape[:2]
    good = []
    for i, (st, pt) in enumerate(zip(status.ravel(), pts2.reshape(-1, 2))):
        if st == 1 and 0 <= pt[0] < w and 0 <= pt[1] < h:
            good.append(i)

    if len(good) < 8:
        return None, None

    good = np.array(good)
    return pts1.reshape(-1, 2)[good], pts2.reshape(-1, 2)[good]


def needs_redetection(pts):
    """True when tracked point count has dropped too low."""
    return pts is None or len(pts) < MIN_FEATURES


def estimate_pose(pts1, pts2, K):
    """
    Compute Essential matrix (RANSAC) and recover relative R, t.

    Returns:
        R, t       – rotation (3,3) and translation (3,1) of current frame
                     relative to previous frame
        pts1m, pts2m – RANSAC inlier correspondences
        Returns (None, None, None, None) on failure.
    """
    E, mask = cv.findEssentialMat(pts2, pts1, K,
                                   cv.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask is None:
        return None, None, None, None

    pts1m = pts1[mask.ravel() == 1]
    pts2m = pts2[mask.ravel() == 1]

    if len(pts1m) < 8:
        return None, None, None, None

    _, R, t, _ = cv.recoverPose(E, pts1m, pts2m, K)
    return R, t, pts1m, pts2m