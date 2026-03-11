"""
visualization.py
────────────────
  • draw_preview   – side-by-side feature-match strip (left frame | right frame)
  • TrajectoryMap  – fast OpenCV-drawn top-down trajectory, updates every frame
  • build_display  – stitches both panels into one window
  • save_plot      – final high-res matplotlib 3-D PNG saved at the end
"""

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# 1. Feature-match preview strip  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def draw_preview(img_prev, img_curr, pts1, pts2,
                 frame_idx, reproj, scale, accepted):
    h, w   = img_curr.shape
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w]  = cv.cvtColor(img_prev, cv.COLOR_GRAY2BGR)
    canvas[:, w:]  = cv.cvtColor(img_curr, cv.COLOR_GRAY2BGR)

    step = max(1, len(pts1) // 80)
    for p1, p2 in zip(pts1[::step], pts2[::step]):
        x1, y1 = int(p1[0]),      int(p1[1])
        x2, y2 = int(p2[0]) + w,  int(p2[1])
        cv.line(canvas,   (x1, y1), (x2, y2), (0, 255, 0),   1)
        cv.circle(canvas, (x2, y2), 2,          (0, 100, 255), -1)

    col = (0, 200, 0) if accepted else (0, 0, 255)
    cv.putText(canvas,
               f"Frame {frame_idx}  [{'OK' if accepted else 'SKIP'}]  "
               f"reproj={reproj:.0f}px  scale={scale:.2f}",
               (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# 2. Fast OpenCV trajectory map
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryMap:
    """
    Draws the camera path directly with OpenCV lines — no matplotlib, no lag.

    The map auto-scales: the view re-centres and zooms whenever the trajectory
    walks near the edge of the canvas.

    Usage:
        tm = TrajectoryMap(width=1280, height=480)
        tm.update(trans)          # call every frame with the (3,) translation
        panel = tm.render()       # → (H, W, 3) BGR uint8, ready for imshow
    """

    def __init__(self, width=1280, height=480):
        self.W = width
        self.H = height
        self._canvas   = np.zeros((height, width, 3), dtype=np.uint8)
        self._pts: list[tuple[float, float]] = []   # (world_x, world_z) pairs
        self._scale  = 1.0      # pixels per world unit
        self._ox     = width  // 2   # canvas origin x (pixels)
        self._oz     = height // 2   # canvas origin z (pixels)

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, trans):
        """
        Add the current camera position and redraw.
        trans – (3,) world translation [x, y, z]
        """
        wx, wz = float(trans[0]), float(trans[2])
        self._pts.append((wx, wz))
        self._refit()
        self._redraw()

    def render(self):
        return self._canvas

    # ── internals ─────────────────────────────────────────────────────────────

    def _world_to_px(self, wx, wz):
        px = int(self._ox + wx * self._scale)
        pz = int(self._oz - wz * self._scale)   # Z grows upward on screen
        return px, pz

    def _refit(self):
        """Recompute scale and origin so all points fit with 15 % padding."""
        if len(self._pts) < 2:
            return
        xs = [p[0] for p in self._pts]
        zs = [p[1] for p in self._pts]
        span_x = max(max(xs) - min(xs), 1e-3)
        span_z = max(max(zs) - min(zs), 1e-3)
        pad    = 1.30                           # 15 % padding on each side
        self._scale = min((self.W * 0.85) / (span_x * pad),
                          (self.H * 0.85) / (span_z * pad))
        cx = (max(xs) + min(xs)) / 2
        cz = (max(zs) + min(zs)) / 2
        self._ox = int(self.W / 2 - cx * self._scale)
        self._oz = int(self.H / 2 + cz * self._scale)

    def _redraw(self):
        canvas = np.full((self.H, self.W, 3), 15, dtype=np.uint8)  # dark bg

        # grid lines
        for v in range(-200, 201, 10):
            gx0, gz0 = self._world_to_px(v, -200)
            gx1, gz1 = self._world_to_px(v,  200)
            cv.line(canvas, (gx0, gz0), (gx1, gz1), (30, 30, 30), 1)
            gx0, gz0 = self._world_to_px(-200, v)
            gx1, gz1 = self._world_to_px( 200, v)
            cv.line(canvas, (gx0, gz0), (gx1, gz1), (30, 30, 30), 1)

        # trajectory line
        for i in range(1, len(self._pts)):
            p0 = self._world_to_px(*self._pts[i - 1])
            p1 = self._world_to_px(*self._pts[i])
            # colour shifts green → cyan as we progress
            t  = i / max(len(self._pts) - 1, 1)
            colour = (0, int(180 + 75 * t), int(255 * (1 - t)))
            cv.line(canvas, p0, p1, colour, 2, cv.LINE_AA)

        # start marker
        sx, sz = self._world_to_px(*self._pts[0])
        cv.circle(canvas, (sx, sz), 6, (0, 255, 80),  -1)
        cv.putText(canvas, "START", (sx + 8, sz + 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 80), 1)

        # current position marker
        cx, cz = self._world_to_px(*self._pts[-1])
        cv.circle(canvas, (cx, cz), 7, (0, 100, 255), -1)
        cv.circle(canvas, (cx, cz), 7, (255, 255, 255), 1)  # white ring

        # HUD
        cv.putText(canvas,
                   f"Trajectory  |  pts={len(self._pts)}  "
                   f"scale=1px:{1/max(self._scale,1e-3):.2f}m",
                   (10, self.H - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        self._canvas = canvas


# ──────────────────────────────────────────────────────────────────────────────
# 3. Stitch panels into one window
# ──────────────────────────────────────────────────────────────────────────────

def build_display(feature_canvas, traj_panel):
    """
    Stack feature strip (top) and trajectory map (bottom).
    Pads width if they differ.
    """
    h1, w1 = feature_canvas.shape[:2]
    h2, w2 = traj_panel.shape[:2]
    W = max(w1, w2)

    def pad(img, w):
        if img.shape[1] == w:
            return img
        p = np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)
        return np.hstack([img, p])

    return np.vstack([pad(feature_canvas, W), pad(traj_panel, W)])


# ──────────────────────────────────────────────────────────────────────────────
# 4. Final high-res 3-D matplotlib PNG  (runs once at the end)
# ──────────────────────────────────────────────────────────────────────────────

def save_plot(traj_x, traj_y, traj_z, cloud_world, out_path="vo_result.png"):
    fig  = plt.figure(figsize=(14, 6))
    ax_t = fig.add_subplot(121, projection='3d')
    ax_c = fig.add_subplot(122, projection='3d')

    ax_t.plot(traj_x, traj_y, traj_z, 'b-', linewidth=0.8)
    ax_t.scatter([traj_x[0]], [traj_y[0]], [traj_z[0]], c='green', s=60, label='start')
    ax_t.scatter([traj_x[-1]], [traj_y[-1]], [traj_z[-1]], c='red', s=60, label='end')
    ax_t.set_title("Camera Trajectory")
    ax_t.set_xlabel("X"); ax_t.set_ylabel("Y"); ax_t.set_zlabel("Z")
    ax_t.legend()

    if cloud_world:
        cp  = np.array(cloud_world)
        idx = np.random.choice(len(cp), min(len(cp), 5000), replace=False)
        ax_c.scatter(cp[idx, 0], cp[idx, 1], cp[idx, 2],
                     s=0.5, c=cp[idx, 2], cmap='viridis', alpha=0.7)
    ax_c.set_title(f"Sparse Point Cloud  ({len(cloud_world)} pts)")
    ax_c.set_xlabel("X"); ax_c.set_ylabel("Y"); ax_c.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved → {out_path}")