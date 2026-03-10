"""
Monocular Visual Odometry — TUM Mono dataset or video

Usage:
  python vo_video_live.py --images dataset/sequence_38/images --calib_file dataset/sequence_38/camera.txt
  python vo_video_live.py --images dataset/sequence_38/images --calib_file dataset/sequence_38/camera.txt --skip 2
"""

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')           # render to file — avoids macOS GUI freeze
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse, os, sys

PRESETS = {
    "kitti":      np.array([[718.856,0.,607.193],[0.,718.856,185.216],[0.,0.,1.]]),
    "custom_cam": np.array([[518.567,0.,329.458],[0.,518.805,237.056],[0.,0.,1.]]),
}

# ──────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────
def load_tum_calib(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    fx_n, fy_n, cx_n, cy_n, omega = map(float, lines[0].split())
    out_w, out_h = map(int, lines[3].split())
    K = np.array([[fx_n*out_w, 0.,         cx_n*out_w],
                  [0.,         fy_n*out_h, cy_n*out_h],
                  [0.,         0.,         1.        ]])
    print(f"[CALIB] fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
          f"cx={K[0,2]:.2f} cy={K[1,2]:.2f} omega={omega:.3f} out={out_w}x{out_h}")
    return K, omega, (out_w, out_h)

# ──────────────────────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────────────────────
class ImageFolderSource:
    def __init__(self, folder, out_size=(640,480)):
        exts = ('.jpg','.jpeg','.png','.bmp')
        self.paths = sorted([os.path.join(folder,f) for f in os.listdir(folder)
                             if f.lower().endswith(exts)])
        self.out_size = out_size
        self.idx = 0
        print(f"[SOURCE] {len(self.paths)} images  {folder}")
    def read(self):
        if self.idx >= len(self.paths): return None
        img = cv.imread(self.paths[self.idx], cv.IMREAD_GRAYSCALE)
        self.idx += 1
        if img is None: return None
        if (img.shape[1], img.shape[0]) != self.out_size:
            img = cv.resize(img, self.out_size)
        return img
    def release(self): pass

class VideoSource:
    def __init__(self, path, out_size=(640,480)):
        self.cap = cv.VideoCapture(path)
        if not self.cap.isOpened(): sys.exit(f"[ERROR] Cannot open {path}")
        self.out_size = out_size
        print(f"[SOURCE] Video {path}  ({int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))} frames)")
    def read(self):
        ret, frame = self.cap.read()
        if not ret: return None
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if (gray.shape[1], gray.shape[0]) != self.out_size:
            gray = cv.resize(gray, self.out_size)
        return gray
    def release(self): self.cap.release()

# ──────────────────────────────────────────────────────────────
# Core VO
# ──────────────────────────────────────────────────────────────
def track_features(img1, img2, n_features=1500):
    sift = cv.SIFT_create(nfeatures=n_features)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(kp1)<8 or len(kp2)<8:
        return None, None
    des1, des2 = np.float32(des1), np.float32(des2)
    flann   = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    pts1, pts2 = [], []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    if len(pts1) < 8: return None, None
    return np.float32(pts1), np.float32(pts2)

def triangulate(R, t, pt1, pt2, K):
    Pr0 = K @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=np.float64)
    Pr1 = K @ np.hstack((R, t.reshape(3,1)))
    return cv.triangulatePoints(Pr0, Pr1, pt1.T, pt2.T)  # (4,N)

def hom_to_3d(cloud4):
    w   = cloud4[3,:] + 1e-8
    xyz = cloud4[:3,:] / w
    mask = (xyz[2,:] > 0.1) & (xyz[2,:] < 200)
    return xyz, mask

def relative_scale(c_old, c_new):
    s = min(c_old.shape[1], c_new.shape[1])
    o, n = c_old[:,:s], c_new[:,:s]
    sc = (np.linalg.norm(o - np.roll(o,1,1), axis=0) /
          (np.linalg.norm(n - np.roll(n,1,1), axis=0) + 1e-8))
    return float(np.median(sc))

def reprojection_error(pts2, cloud4, R_abs, t_abs, K):
    xyz, mask = hom_to_3d(cloud4)
    if mask.sum() < 4: return float('inf')
    pts2_ok = pts2.reshape(-1,2)[mask]   # (M,2)
    xyz_ok  = xyz[:,mask]                # (3,M)
    P  = np.hstack((R_abs, t_abs.reshape(3,1)))
    Xh = np.vstack((xyz_ok, np.ones((1,xyz_ok.shape[1]))))
    proj = K @ (P @ Xh)
    proj = (proj[:2,:] / (proj[2:3,:] + 1e-8)).T  # (M,2)
    return float(np.mean(np.linalg.norm(proj - pts2_ok, axis=1)))

# ──────────────────────────────────────────────────────────────
# Draw matches overlay for the preview window
# ──────────────────────────────────────────────────────────────
def draw_preview(img_prev, img_curr, pts1, pts2, frame_idx,
                 reproj, scale, accepted):
    # side-by-side
    h, w = img_curr.shape
    canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
    canvas[:, :w]  = cv.cvtColor(img_prev, cv.COLOR_GRAY2BGR)
    canvas[:, w:]  = cv.cvtColor(img_curr, cv.COLOR_GRAY2BGR)

    # draw flow lines (subsample for speed)
    step = max(1, len(pts1)//80)
    for p1, p2 in zip(pts1[::step], pts2[::step]):
        x1,y1 = int(p1[0]),   int(p1[1])
        x2,y2 = int(p2[0])+w, int(p2[1])
        cv.line(canvas, (x1,y1), (x2,y2), (0,255,0), 1)
        cv.circle(canvas, (x2,y2), 2, (0,100,255), -1)

    status_col = (0,200,0) if accepted else (0,0,255)
    status_txt = "OK" if accepted else "SKIP"
    cv.putText(canvas, f"Frame {frame_idx}  [{status_txt}]  "
               f"reproj={reproj:.0f}px  scale={scale:.2f}",
               (10,25), cv.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)
    return canvas

# ──────────────────────────────────────────────────────────────
# Save trajectory plot to PNG
# ──────────────────────────────────────────────────────────────
def save_plot(traj_x, traj_y, traj_z, cloud_world, out_path="vo_result.png"):
    fig = plt.figure(figsize=(14,6))
    ax_t = fig.add_subplot(121, projection='3d')
    ax_c = fig.add_subplot(122, projection='3d')

    ax_t.plot(traj_x, traj_y, traj_z, 'b-', linewidth=0.8)
    ax_t.scatter([traj_x[0]],[traj_y[0]],[traj_z[0]], c='green', s=60, label='start')
    ax_t.scatter([traj_x[-1]],[traj_y[-1]],[traj_z[-1]], c='red', s=60, label='end')
    ax_t.set_title("Camera Trajectory")
    ax_t.set_xlabel("X"); ax_t.set_ylabel("Y"); ax_t.set_zlabel("Z")
    ax_t.legend()

    if len(cloud_world) > 0:
        cp  = np.array(cloud_world)
        idx = np.random.choice(len(cp), min(len(cp),5000), replace=False)
        ax_c.scatter(cp[idx,0], cp[idx,1], cp[idx,2],
                     s=0.5, c=cp[idx,2], cmap='viridis', alpha=0.7)
    ax_c.set_title(f"Sparse Point Cloud  ({len(cloud_world)} pts)")
    ax_c.set_xlabel("X"); ax_c.set_ylabel("Y"); ax_c.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved → {out_path}")

# ──────────────────────────────────────────────────────────────
# Main VO loop
# ──────────────────────────────────────────────────────────────
def run(source, K, skip=1, max_frames=3000,
        reproj_thresh=500.0, plot_every=10, show_preview=True):
    """
    reproj_thresh: pixels. Fisheye images have large distortion so
                   we use a relaxed default (500px). The reproj error
                   here is computed with the pinhole K applied to
                   distorted images, so it will always be inflated.
                   The value is still useful as a relative outlier filter.
    """

    # ── bootstrap ──────────────────────────────────────────
    img_prev = source.read()
    if img_prev is None: sys.exit("[ERROR] Empty source.")
    img_curr = source.read()

    pts1, pts2 = None, None
    while pts1 is None:
        pts1, pts2 = track_features(img_prev, img_curr)
        if pts1 is None:
            img_prev = img_curr
            img_curr = source.read()
            if img_curr is None: sys.exit("[ERROR] Not enough frames.")

    E, mask = cv.findEssentialMat(pts2, pts1, K, cv.RANSAC, prob=0.999, threshold=1.0)
    pts1 = pts1[mask.ravel()==1];  pts2 = pts2[mask.ravel()==1]
    _, R, t, _ = cv.recoverPose(E, pts1, pts2, K)

    n_cloud4 = triangulate(np.eye(3), np.zeros(3), pts1, pts2, K)
    rotation = R.copy()
    trans    = t.flatten()          # always keep as (3,)

    traj_x = [0.0, float(trans[0])]
    traj_y = [0.0, float(trans[1])]
    traj_z = [0.0, float(trans[2])]

    cloud_world = []
    xyz0, m0 = hom_to_3d(n_cloud4)
    cloud_world.extend(xyz0[:,m0].T.tolist())

    frame_idx   = 1
    consec_fail = 0
    accepted_total = 0

    print(f"[INFO] reproj_thresh={reproj_thresh}px   "
          f"(fisheye → values will be large, that's expected)")
    print("[INFO] OpenCV preview window will open. Press Q to quit early.\n")

    while frame_idx < max_frames:

        # ── advance ────────────────────────────────────────
        img_next = None
        for _ in range(skip):
            img_next = source.read()
            if img_next is None: break
        if img_next is None:
            print("[INFO] End of sequence."); break

        # ── features ───────────────────────────────────────
        pts1, pts2 = track_features(img_curr, img_next)
        if pts1 is None:
            img_curr = img_next; frame_idx += 1
            consec_fail += 1
            if consec_fail > 15: print("[ERROR] Too many failures."); break
            continue
        consec_fail = 0

        # ── essential matrix ───────────────────────────────
        E, mask = cv.findEssentialMat(pts2, pts1, K, cv.RANSAC,
                                       prob=0.999, threshold=1.0)
        if E is None:
            img_curr = img_next; frame_idx += 1; continue
        pts1m = pts1[mask.ravel()==1];  pts2m = pts2[mask.ravel()==1]
        if len(pts1m) < 8:
            img_curr = img_next; frame_idx += 1; continue

        _, R, t, _ = cv.recoverPose(E, pts1m, pts2m, K)
        t_vec = t.flatten()

        # ── triangulate + scale ────────────────────────────
        o_cloud4 = n_cloud4
        n_cloud4 = triangulate(R, t_vec, pts1m, pts2m, K)
        o3d, _ = hom_to_3d(o_cloud4)
        n3d, _ = hom_to_3d(n_cloud4)
        sc = np.clip(relative_scale(o3d, n3d), 0.05, 20.0)

        # ── pose accumulation ──────────────────────────────
        trans    = trans - sc*(rotation @ t_vec)
        rotation = rotation @ R

        rep_err  = reprojection_error(pts2m, n_cloud4, rotation, trans, K)
        accepted = rep_err < reproj_thresh

        if accepted:
            traj_x.append(float(trans[0]))
            traj_y.append(float(trans[1]))
            traj_z.append(float(trans[2]))
            n3d_local, valid = hom_to_3d(n_cloud4)
            wpts = (rotation @ n3d_local[:,valid]) + trans.reshape(3,1)
            cloud_world.extend(wpts.T.tolist())
            accepted_total += 1

        # ── OpenCV preview ─────────────────────────────────
        if show_preview:
            preview = draw_preview(img_prev, img_curr, pts1m, pts2m,
                                   frame_idx, rep_err, sc, accepted)
            cv.imshow("VO Preview  (Q to quit)", preview)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] User quit."); break

        # ── terminal log ───────────────────────────────────
        if frame_idx % plot_every == 0:
            tag = "OK  " if accepted else "SKIP"
            print(f"  [{tag}] Frame {frame_idx:4d} | "
                  f"feats={len(pts1m):3d} | scale={sc:.3f} | "
                  f"reproj={rep_err:.0f}px | "
                  f"accepted={accepted_total} | cloud={len(cloud_world)}")

        img_prev = img_curr
        img_curr = img_next
        frame_idx += 1

    source.release()
    cv.destroyAllWindows()

    # ── save final plot ────────────────────────────────────
    if len(traj_x) > 2:
        save_plot(traj_x, traj_y, traj_z, cloud_world)
    else:
        print("[WARN] Not enough accepted frames to plot trajectory.")
        print("       Try increasing --reproj_thresh (e.g. 2000)")

    print(f"\n[DONE] {frame_idx} frames | "
          f"{accepted_total} accepted | {len(cloud_world)} cloud pts")
    return traj_x, traj_y, traj_z, cloud_world

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
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
    p.add_argument("--skip",          type=int,   default=1)
    p.add_argument("--max_frames",    type=int,   default=3000)
    p.add_argument("--reproj_thresh", type=float, default=500.0,
                   help="Reprojection error threshold in pixels. "
                        "Use large values (500-2000) for fisheye images.")
    p.add_argument("--plot_every",    type=int,   default=10)
    p.add_argument("--no_preview",    action="store_true",
                   help="Disable the OpenCV preview window")
    p.add_argument("--out_w", type=int, default=640)
    p.add_argument("--out_h", type=int, default=480)
    args = p.parse_args()

    out_size = (args.out_w, args.out_h)
    if args.calib_file:
        K, _, out_size = load_tum_calib(args.calib_file)
    elif args.calib:
        K = PRESETS[args.calib]; print(f"[CALIB] preset={args.calib}")
    else:
        K = np.array([[args.fx,0.,args.cx],[0.,args.fy,args.cy],[0.,0.,1.]])

    source = (ImageFolderSource(args.images, out_size) if args.images
              else VideoSource(args.video, out_size))

    run(source, K,
        skip=args.skip,
        max_frames=args.max_frames,
        reproj_thresh=args.reproj_thresh,
        plot_every=args.plot_every,
        show_preview=not args.no_preview)