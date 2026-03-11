"""
main.py  –  Monocular Visual Odometry Pipeline
───────────────────────────────────────────────
Pipeline stages:
  Stage 0 │ calibration.py      – load / build intrinsic matrix K
  Stage 1 │ sources.py          – stream grayscale frames
  Stage 2 │ feature_tracking.py – SIFT + FLANN, Essential matrix, pose
  Stage 3 │ geometry.py         – triangulation, scale, reprojection error
  Stage 4 │ visualization.py    – live unified window (video + trajectory)
                                   + final matplotlib PNG

Usage:
  python main.py --images dataset/sequence_38/images --calib_file dataset/sequence_38/camera.txt
  python main.py --video  my_drive.mp4 --calib kitti --skip 2 --no_preview
"""

import sys, argparse
import numpy as np
import cv2 as cv

from calibration      import build_K_from_args, PRESETS
from sources          import build_source
from feature_tracking import track_features, estimate_pose
from geometry         import triangulate, hom_to_3d, relative_scale, reprojection_error
from visualization    import draw_preview, TrajectoryMap, build_display, save_plot


# ─────────────────────────────────────────────────────────────────────────────
def run(source, K, skip=1, max_frames=3000,
        reproj_thresh=500.0, plot_every=10, show_preview=True):

    # ── bootstrap ────────────────────────────────────────────────────────────
    img_prev = source.read()
    if img_prev is None:
        sys.exit("[ERROR] Empty source.")
    img_curr = source.read()

    pts1, pts2 = None, None
    while pts1 is None:
        pts1, pts2 = track_features(img_prev, img_curr)
        if pts1 is None:
            img_prev = img_curr
            img_curr = source.read()
            if img_curr is None:
                sys.exit("[ERROR] Not enough frames to bootstrap.")

    R, t, pts1, pts2 = estimate_pose(pts1, pts2, K)
    if R is None:
        sys.exit("[ERROR] Could not estimate initial pose.")

    n_cloud4 = triangulate(np.eye(3), np.zeros(3), pts1, pts2, K)
    rotation = R.copy()
    trans    = t.flatten()

    traj_x = [0.0, float(trans[0])]
    traj_y = [0.0, float(trans[1])]
    traj_z = [0.0, float(trans[2])]

    cloud_world = []
    xyz0, m0    = hom_to_3d(n_cloud4)
    cloud_world.extend(xyz0[:, m0].T.tolist())

    # ── live plot panel ───────────────────────────────────────────────────────
    frame_w = img_curr.shape[1]
    # feature strip is frame_w*2 wide; match that for the plot panel
    traj_map = TrajectoryMap(width=frame_w * 2, height=360)
    traj_map.update(trans)

    frame_idx      = 1
    consec_fail    = 0
    accepted_total = 0

    print(f"[INFO] reproj_thresh={reproj_thresh} px")
    if show_preview:
        print("[INFO] Combined window open (video top, trajectory bottom). "
              "Press Q to quit early.\n")

    # ── main loop ─────────────────────────────────────────────────────────────
    while frame_idx < max_frames:

        img_next = None
        for _ in range(skip):
            img_next = source.read()
            if img_next is None:
                break
        if img_next is None:
            print("[INFO] End of sequence.")
            break

        # Stage 2 – feature tracking + pose
        pts1_raw, pts2_raw = track_features(img_curr, img_next)
        if pts1_raw is None:
            img_curr = img_next; frame_idx += 1
            consec_fail += 1
            if consec_fail > 15:
                print("[ERROR] Too many consecutive failures.")
                break
            continue
        consec_fail = 0

        R, t, pts1m, pts2m = estimate_pose(pts1_raw, pts2_raw, K)
        if R is None:
            img_curr = img_next; frame_idx += 1; continue
        t_vec = t.flatten()

        # Stage 3 – triangulate + scale
        o_cloud4 = n_cloud4
        n_cloud4 = triangulate(R, t_vec, pts1m, pts2m, K)
        o3d, _   = hom_to_3d(o_cloud4)
        n3d, _   = hom_to_3d(n_cloud4)
        sc       = np.clip(relative_scale(o3d, n3d), 0.05, 20.0)

        trans    = trans - sc * (rotation @ t_vec)
        rotation = rotation @ R

        rep_err  = reprojection_error(pts2m, n_cloud4, rotation, trans, K)
        accepted = rep_err < reproj_thresh

        if accepted:
            traj_x.append(float(trans[0]))
            traj_y.append(float(trans[1]))
            traj_z.append(float(trans[2]))
            n3d_local, valid = hom_to_3d(n_cloud4)
            wpts = (rotation @ n3d_local[:, valid]) + trans.reshape(3, 1)
            cloud_world.extend(wpts.T.tolist())
            accepted_total += 1

            # refresh the live plot every accepted frame
            traj_map.update(trans)

        # Stage 4 – combined display
        if show_preview:
            feature_canvas = draw_preview(img_prev, img_curr, pts1m, pts2m,
                                          frame_idx, rep_err, sc, accepted)
            display = build_display(feature_canvas, traj_map.render())
            cv.imshow("VO Preview  (Q to quit)", display)
            if (cv.waitKey(1) & 0xFF) == ord('q'):
                print("[INFO] User quit.")
                break

        if frame_idx % plot_every == 0:
            tag = "OK  " if accepted else "SKIP"
            print(f"  [{tag}] Frame {frame_idx:4d} | feats={len(pts1m):3d} | "
                  f"scale={sc:.3f} | reproj={rep_err:.0f} px | "
                  f"accepted={accepted_total} | cloud={len(cloud_world)}")

        img_prev = img_curr
        img_curr = img_next
        frame_idx += 1

    source.release()
    cv.destroyAllWindows()

    if len(traj_x) > 2:
        save_plot(traj_x, traj_y, traj_z, cloud_world)
    else:
        print("[WARN] Not enough accepted frames to plot.")
        print("       Try increasing --reproj_thresh (e.g. 2000)")

    print(f"\n[DONE] {frame_idx} frames | "
          f"{accepted_total} accepted | {len(cloud_world)} cloud pts")
    return traj_x, traj_y, traj_z, cloud_world


# ─────────────────────────────────────────────────────────────────────────────
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
    p.add_argument("--out_w",         type=int,   default=640)
    p.add_argument("--out_h",         type=int,   default=480)
    p.add_argument("--skip",          type=int,   default=1)
    p.add_argument("--max_frames",    type=int,   default=3000)
    p.add_argument("--reproj_thresh", type=float, default=500.0)
    p.add_argument("--plot_every",    type=int,   default=10)
    p.add_argument("--no_preview",    action="store_true")
    return p


if __name__ == "__main__":
    args        = build_parser().parse_args()
    K, out_size = build_K_from_args(args)
    source      = build_source(args, out_size)
    run(source, K,
        skip          = args.skip,
        max_frames    = args.max_frames,
        reproj_thresh = args.reproj_thresh,
        plot_every    = args.plot_every,
        show_preview  = not args.no_preview)