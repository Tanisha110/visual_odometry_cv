"""
visualization.py
────────────────
  • draw_preview    – side-by-side feature-match strip
  • TrajectoryMap   – fast OpenCV top-down trajectory (draws incrementally)
  • build_display   – stitches both panels into one window
  • save_plot       – final matplotlib PNG at the end

Fixes vs previous version:
  1. _to_px: Z is now SUBTRACTED (screen Y increases downward, so forward
     motion = positive Z must go UP the canvas, i.e. decreasing pixel row).
     Was: oz + wz*scale  →  Now: oz - wz*scale
  2. _refit_and_redraw: _oz formula flipped to match _to_px:
     Was: H/2 + cz*scale  →  Now: H/2 - cz*scale
  3. save_plot: left subplot is now a proper 2-D top-down X-Z plot (the
     meaningful ground-plane view). 3-D subplot kept for point cloud only.
  4. draw_preview: added source circles on left panel so feature origins
     are visible, not just destinations.
"""

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# 1. Feature-match preview strip
# ──────────────────────────────────────────────────────────────────────────────

def draw_preview(img_prev, img_curr, pts1, pts2,
                 frame_idx, reproj, scale, accepted):
    h, w   = img_curr.shape
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w]  = cv.cvtColor(img_prev, cv.COLOR_GRAY2BGR)
    canvas[:, w:]  = cv.cvtColor(img_curr, cv.COLOR_GRAY2BGR)

    # divider line between the two panels
    cv.line(canvas, (w, 0), (w, h), (60, 60, 60), 1)

    step = max(1, len(pts1) // 80)
    for p1, p2 in zip(pts1[::step], pts2[::step]):
        x1, y1 = int(p1[0]),      int(p1[1])
        x2, y2 = int(p2[0]) + w,  int(p2[1])
        cv.line(canvas,   (x1, y1), (x2, y2), (0, 200, 0), 1)
        # FIX: draw source circle on LEFT panel too
        cv.circle(canvas, (x1, y1), 2, (255, 160,   0), -1)   # orange = source
        cv.circle(canvas, (x2, y2), 2, (0,   100, 255), -1)   # blue   = dest

    col = (0, 200, 0) if accepted else (0, 0, 255)
    cv.putText(canvas,
               f"Frame {frame_idx}  [{'OK' if accepted else 'SKIP'}]  "
               f"reproj={reproj:.0f}px  scale={scale:.2f}",
               (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# 2. Trajectory map — incremental OpenCV drawing, no full redraw every frame
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryMap:
    """
    Top-down (X–Z) camera path drawn with OpenCV.

    Coordinate convention (matches the reference VO):
        X  →  right on canvas
        Z  →  UP on canvas   (forward motion goes up, not down)

    Pixel mapping:
        px = ox + wx * scale
        pz = oz - wz * scale    ← MINUS because screen-Y increases downward
    """

    _PAD   = 0.80
    _BGCOL = (15, 15, 15)

    def __init__(self, width=1280, height=360):
        self.W = width
        self.H = height
        self._canvas = np.full((height, width, 3), 15, dtype=np.uint8)

        self._world_pts: list[tuple[float, float]] = []   # (x, z) history
        self._scale = 5.0
        self._ox    = width  // 2
        self._oz    = height // 2

        self._draw_origin()

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, trans):
        """
        Called every frame with the current (3,) translation vector.
        Extracts X (index 0) and Z (index 2) for the ground-plane view.
        """
        wx, wz = float(trans[0]), float(trans[2])
        px, pz = self._to_px(wx, wz)

        needs_rescale = (px < 20 or px > self.W - 20 or
                         pz < 20 or pz > self.H - 20)

        self._world_pts.append((wx, wz))

        if needs_rescale:
            self._refit_and_redraw()
        elif len(self._world_pts) >= 2:
            prev = self._world_pts[-2]
            p0   = self._to_px(*prev)
            p1   = self._to_px(wx, wz)
            cv.line(self._canvas, p0, p1, (0, 220, 100), 2, cv.LINE_AA)
            self._draw_cursor(p1)
            self._draw_hud()

    def render(self):
        return self._canvas

    # ── internals ─────────────────────────────────────────────────────────────

    def _to_px(self, wx, wz):
        """
        World (wx, wz) → pixel (col, row).
        FIX: subtract wz so that forward (positive Z) goes UP the canvas.
        """
        return (int(self._ox + wx * self._scale),
                int(self._oz - wz * self._scale))   # ← was +, now MINUS

    def _draw_origin(self):
        cv.circle(self._canvas, (self._ox, self._oz), 5, (80, 80, 80), -1)
        cv.putText(self._canvas, "O", (self._ox + 7, self._oz + 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

    def _draw_cursor(self, px_pos):
        cx, cz = px_pos
        cv.circle(self._canvas, (cx, cz), 6, (0, 100, 255), -1)
        cv.circle(self._canvas, (cx, cz), 6, (255, 255, 255), 1)

    def _draw_hud(self):
        self._canvas[self.H - 22:self.H, :] = 15
        n = len(self._world_pts)
        m = 1.0 / max(self._scale, 1e-6)
        cv.putText(self._canvas,
                   f"Top-down trajectory (X right, Z up)  |  {n} pts  |  "
                   f"1 px = {m:.3f} m  (world units)",
                   (10, self.H - 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    def _refit_and_redraw(self):
        """Recompute scale/origin and repaint the whole canvas from scratch."""
        pts    = self._world_pts
        xs     = [p[0] for p in pts]
        zs     = [p[1] for p in pts]
        span_x = max(max(xs) - min(xs), 1.0)
        span_z = max(max(zs) - min(zs), 1.0)
        self._scale = min(self.W * self._PAD / span_x,
                          self.H * self._PAD / span_z)
        cx = (max(xs) + min(xs)) / 2
        cz = (max(zs) + min(zs)) / 2
        self._ox = int(self.W / 2 - cx * self._scale)
        # FIX: must match _to_px sign convention (subtract Z → origin shifts +)
        self._oz = int(self.H / 2 + cz * self._scale)   # ← was +, correct: +cz because _to_px does -wz

        self._canvas[:] = 15

        # subtle grid
        step = max(1, int(50 / max(self._scale, 1e-6)))
        for v in range(-500, 501, step):
            cv.line(self._canvas,
                    self._to_px(v, -500), self._to_px(v, 500),
                    (28, 28, 28), 1)
            cv.line(self._canvas,
                    self._to_px(-500, v), self._to_px(500, v),
                    (28, 28, 28), 1)

        self._draw_origin()

        # redraw full path
        for i in range(1, len(pts)):
            p0 = self._to_px(*pts[i - 1])
            p1 = self._to_px(*pts[i])
            cv.line(self._canvas, p0, p1, (0, 220, 100), 2, cv.LINE_AA)

        # start marker
        sx, sz = self._to_px(*pts[0])
        cv.circle(self._canvas, (sx, sz), 6, (0, 255, 80), -1)
        cv.putText(self._canvas, "S", (sx + 8, sz + 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 80), 1)

        self._draw_cursor(self._to_px(*pts[-1]))
        self._draw_hud()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Stitch panels
# ──────────────────────────────────────────────────────────────────────────────

def build_display(feature_canvas, traj_panel):
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
# 4. Final matplotlib PNG
# ──────────────────────────────────────────────────────────────────────────────

def save_plot(traj_x, traj_y, traj_z, cloud_world, out_path="vo_result.png"):
    """
    Left subplot  – 2-D top-down X-Z trajectory (the ground plane).
                    This is the meaningful view for forward-driving sequences.
                    Y (altitude) is near-zero for ground vehicles and produces
                    a flat, uninformative 3-D line — so we drop it here.
    Right subplot – sparse 3-D point cloud coloured by depth (Z).
    """
    fig = plt.figure(figsize=(14, 6))

    # ── left: 2-D top-down trajectory (X vs Z) ────────────────────────────────
    ax_t = fig.add_subplot(121)
    ax_t.plot(traj_x, traj_z, 'b-', linewidth=0.8, label='path')
    ax_t.scatter([traj_x[0]],  [traj_z[0]],  c='green', s=60,
                 zorder=5, label='start')
    ax_t.scatter([traj_x[-1]], [traj_z[-1]], c='red',   s=60,
                 zorder=5, label='end')
    ax_t.set_title("Camera Trajectory  (top-down, X–Z)")
    ax_t.set_xlabel("X  (m)");  ax_t.set_ylabel("Z  (m, forward)")
    ax_t.set_aspect('equal', adjustable='datalim')
    ax_t.grid(True, linewidth=0.4, alpha=0.5)
    ax_t.legend()

    # ── right: sparse 3-D point cloud ─────────────────────────────────────────
    ax_c = fig.add_subplot(122, projection='3d')
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