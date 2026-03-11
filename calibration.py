"""
calibration.py
──────────────
Camera intrinsics: load from TUM calib file or use a named preset.
"""

import numpy as np

PRESETS = {
    "kitti":      np.array([[718.856, 0.,      607.193],
                             [0.,      718.856, 185.216],
                             [0.,      0.,      1.     ]]),
    "custom_cam": np.array([[518.567, 0.,      329.458],
                             [0.,      518.805, 237.056],
                             [0.,      0.,      1.     ]]),
}


def load_tum_calib(path):
    """
    Parse a TUM Mono camera.txt file.
    Returns:
        K        – (3,3) intrinsic matrix
        omega    – fisheye distortion parameter (float)
        out_size – (width, height) tuple
    """
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    fx_n, fy_n, cx_n, cy_n, omega = map(float, lines[0].split())
    out_w, out_h = map(int, lines[3].split())
    K = np.array([[fx_n * out_w, 0.,           cx_n * out_w],
                  [0.,           fy_n * out_h,  cy_n * out_h],
                  [0.,           0.,            1.           ]])
    print(f"[CALIB] fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  "
          f"cx={K[0,2]:.2f}  cy={K[1,2]:.2f}  "
          f"omega={omega:.3f}  out={out_w}x{out_h}")
    return K, omega, (out_w, out_h)


def build_K_from_args(args):
    """
    Build / select intrinsic matrix from CLI args.
    Returns K, out_size.
    """
    out_size = (args.out_w, args.out_h)
    if args.calib_file:
        K, _, out_size = load_tum_calib(args.calib_file)
    elif args.calib:
        K = PRESETS[args.calib]
        print(f"[CALIB] preset={args.calib}")
    else:
        K = np.array([[args.fx, 0.,      args.cx],
                      [0.,      args.fy,  args.cy],
                      [0.,      0.,       1.      ]])
    return K, out_size
