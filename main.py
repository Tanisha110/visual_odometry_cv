"""
main.py  –  Monocular Visual Odometry Pipeline
───────────────────────────────────────────────
  Stage 0 │ calibration.py       – intrinsic matrix K
  Stage 1 │ sources.py           – frame streaming
  Stage 2 │ feature_tracking.py  – FAST detect + LK optical flow + pose
  Stage 3 │ geometry.py          – pose accumulation, triangulation, scale
  Stage 4 │ visualization.py     – live window + final PNG

Usage:
  python main.py --images dataset/sequence_38/images --calib_file dataset/sequence_38/camera.txt
  python main.py --video  my_drive.mp4 --calib kitti --skip 2 --no_preview

Fixes applied vs previous version:
  1. geometry.py: R_f = R_f @ R  (post-multiply, was R @ R_f → caused spiral drift)
  2. draw_preview now receives img_curr/img_next (was img_prev/img_curr – wrong pair)
  3. TrajectoryMap.update and save_plot use X,Z (ground plane), not X,Y
  4. relative_scale guard for empty clouds
"""

import sys, argparse
import numpy as np
import cv2 as cv

from calibration      import build_K_from_args, PRESETS
from sources          import build_source
from feature_tracking import detect_features, track_features, \
                              needs_redetection, estimate_pose
from geometry         import pose_update, scale_is_valid, \
                              triangulate, hom_to_3d, relative_scale
from visualization    import draw_preview, TrajectoryMap, build_display, save_plot


def run(source, K, skip=1, max_frames=3000,
        plot_every=10, show_preview=True):

    # ── bootstrap: grab first two frames ─────────────────────────────────────
    img_prev = source.read()
    if img_prev is None:
        sys.exit("[ERROR] Empty source.")
    img_curr = source.read()

    # detect in first frame, track to second
    prev_pts = detect_features(img_prev)
    pts1, pts2 = track_features(img_prev, img_curr, prev_pts)
    while pts1 is None:
        img_prev = img_curr
        img_curr = source.read()
        if img_curr is None:
            sys.exit("[ERROR] Not enough frames to bootstrap.")
        prev_pts = detect_features(img_prev)
        pts1, pts2 = track_features(img_prev, img_curr, prev_pts)

    R, t, pts1, pts2 = estimate_pose(pts1, pts2, K)
    if R is None:
        sys.exit("[ERROR] Could not estimate initial pose.")

    # global pose  (matches C++ R_f, t_f)
    R_f = R.copy()
    t_f = t.flatten().copy()

    # trajectory — store every accepted position  (X and Z = ground plane)
    traj_x = [0.0, float(t_f[0])]
    traj_z = [0.0, float(t_f[2])]   # ← Z is forward/depth, used for 2-D map
    traj_y = [0.0, float(t_f[1])]   # kept for optional 3-D plot

    # point cloud (stored, not drawn live)
    cloud_world: list = []

    # initial triangulation for scale reference
    n_cloud4 = triangulate(np.eye(3), np.zeros(3), pts1, pts2, K)
    curr_pts = pts2.copy()   # points to track in the next iteration

    # ── trajectory display panel ──────────────────────────────────────────────
    frame_w  = img_curr.shape[1]
    traj_map = TrajectoryMap(width=frame_w * 2, height=360)
    traj_map.update(t_f)

    frame_idx      = 1
    consec_fail    = 0
    accepted_total = 0

    print(f"[INFO] Starting VO loop  (FAST + LK flow)")
    if show_preview:
        print("[INFO] Window: feature flow (top) | trajectory (bottom). Q = quit\n")

    # ── main loop ─────────────────────────────────────────────────────────────
    while frame_idx < max_frames:

        # advance by `skip` frames
        img_next = None
        for _ in range(skip):
            img_next = source.read()
            if img_next is None:
                break
        if img_next is None:
            print("[INFO] End of sequence.")
            break

        # ── redetect if too few points are being tracked ──────────────────────
        if needs_redetection(curr_pts):
            curr_pts = detect_features(img_curr)

        # ── LK optical flow tracking ──────────────────────────────────────────
        pts1m, pts2m = track_features(img_curr, img_next, curr_pts)
        if pts1m is None:
            curr_pts    = detect_features(img_next)
            img_prev    = img_curr
            img_curr    = img_next
            frame_idx  += 1
            consec_fail += 1
            if consec_fail > 20:
                print("[ERROR] Too many consecutive tracking failures.")
                break
            continue
        consec_fail = 0

        # ── pose estimation ───────────────────────────────────────────────────
        R, t, pts1m, pts2m = estimate_pose(pts1m, pts2m, K)
        if R is None:
            curr_pts   = detect_features(img_next)
            img_prev   = img_curr
            img_curr   = img_next
            frame_idx += 1
            continue
        t_vec = t.flatten()

        # ── scale recovery from triangulation ────────────────────────────────
        o_cloud4 = n_cloud4
        n_cloud4 = triangulate(R, t_vec, pts1m, pts2m, K)
        o3d, _   = hom_to_3d(o_cloud4)
        n3d, _   = hom_to_3d(n_cloud4)
        sc       = np.clip(relative_scale(o3d, n3d), 0.1, 20.0)

        # ── pose accumulation ─────────────────────────────────────────────────
        accepted = scale_is_valid(t_vec, sc)
        if accepted:
            R_f, t_f = pose_update(R_f, t_f, R, t_vec, sc)
            accepted_total += 1

            # store cloud points in world frame
            n3d_local, valid = hom_to_3d(n_cloud4)
            if valid.any():
                wpts = (R_f @ n3d_local[:, valid]) + t_f.reshape(3, 1)
                cloud_world.extend(wpts.T.tolist())

        # always record trajectory and update map  (X,Z ground plane)
        traj_x.append(float(t_f[0]))
        traj_y.append(float(t_f[1]))
        traj_z.append(float(t_f[2]))
        traj_map.update(t_f)

        # ── display ───────────────────────────────────────────────────────────
        if show_preview:
            # FIX: pass img_curr + img_next (the pair whose matches we just used),
            #      NOT img_prev + img_curr (which is one step behind)
            feature_canvas = draw_preview(img_curr, img_next, pts1m, pts2m,
                                          frame_idx, sc, sc, accepted)
            display = build_display(feature_canvas, traj_map.render())
            cv.imshow("VO  |  Q to quit", display)
            if (cv.waitKey(1) & 0xFF) == ord('q'):
                print("[INFO] User quit.")
                break

        if frame_idx % plot_every == 0:
            tag = "OK  " if accepted else "SKIP"
            print(f"  [{tag}] Frame {frame_idx:4d} | "
                  f"tracked={len(pts1m):4d} | scale={sc:.3f} | "
                  f"pos=({t_f[0]:.1f}, {t_f[1]:.1f}, {t_f[2]:.1f}) | "
                  f"accepted={accepted_total}")

        curr_pts = pts2m.copy()   # carry forward tracked points
        img_prev = img_curr
        img_curr = img_next
        frame_idx += 1

    source.release()
    cv.destroyAllWindows()

    if len(traj_x) > 2:
        save_plot(traj_x, traj_y, traj_z, cloud_world)
    else:
        print("[WARN] Not enough frames processed.")

    print(f"\n[DONE] {frame_idx} frames | "
          f"{accepted_total} accepted | {len(cloud_world)} cloud pts")
    return traj_x, traj_y, traj_z, cloud_world


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(description="Monocular Visual Odometry")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images")
    src.add_argument("--video")
    cal = p.add_mutually_exclusive_group()
    cal.add_argument("--calib_file")
    cal.add_argument("--calib", choices=list(PRESETS.keys()))
    p.add_argument("--fx",  type=float, default=342.86)
    p.add_argument("--fy",  type=float, default=321.39)
    p.add_argument("--cx",  type=float, default=315.68)
    p.add_argument("--cy",  type=float, default=240.20)
    p.add_argument("--out_w",      type=int,   default=640)
    p.add_argument("--out_h",      type=int,   default=480)
    p.add_argument("--skip",       type=int,   default=1)
    p.add_argument("--max_frames", type=int,   default=3000)
    p.add_argument("--plot_every", type=int,   default=10)
    p.add_argument("--no_preview", action="store_true")
    return p


if __name__ == "__main__":
    args        = build_parser().parse_args()
    K, out_size = build_K_from_args(args)
    source      = build_source(args, out_size)
    run(source, K,
        skip         = args.skip,
        max_frames   = args.max_frames,
        plot_every   = args.plot_every,
        show_preview = not args.no_preview)