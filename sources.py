"""
sources.py
──────────
Frame providers: image folder on disk  or  video file.
Both expose the same interface:   .read() → grayscale ndarray | None
                                  .release()
"""

import os, sys
import cv2 as cv


class ImageFolderSource:
    """Reads sorted images from a directory."""

    def __init__(self, folder, out_size=(640, 480)):
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(exts)
        ])
        self.out_size = out_size
        self.idx      = 0
        print(f"[SOURCE] {len(self.paths)} images  ←  {folder}")

    def read(self):
        if self.idx >= len(self.paths):
            return None
        img = cv.imread(self.paths[self.idx], cv.IMREAD_GRAYSCALE)
        self.idx += 1
        if img is None:
            return None
        if (img.shape[1], img.shape[0]) != self.out_size:
            img = cv.resize(img, self.out_size)
        return img

    def release(self):
        pass   # nothing to close


class VideoSource:
    """Reads frames from a video file."""

    def __init__(self, path, out_size=(640, 480)):
        self.cap = cv.VideoCapture(path)
        if not self.cap.isOpened():
            sys.exit(f"[ERROR] Cannot open {path}")
        self.out_size = out_size
        print(f"[SOURCE] Video  {path}  "
              f"({int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))} frames)")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if (gray.shape[1], gray.shape[0]) != self.out_size:
            gray = cv.resize(gray, self.out_size)
        return gray

    def release(self):
        self.cap.release()


def build_source(args, out_size):
    """Factory: pick the right source based on CLI args."""
    if args.images:
        return ImageFolderSource(args.images, out_size)
    return VideoSource(args.video, out_size)
