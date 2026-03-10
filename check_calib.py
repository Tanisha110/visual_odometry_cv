import numpy as np
import cv2 as cv
import os

# --- Parse camera.txt ---
def load_tum_mono_calib(camera_txt_path):
    with open(camera_txt_path) as f:
        lines = f.read().splitlines()
    
    vals = list(map(float, lines[0].split()))
    fx_n, fy_n, cx_n, cy_n, omega = vals
    
    in_w, in_h   = map(int, lines[1].split())
    # line[2] is "crop" or "none"
    out_w, out_h = map(int, lines[3].split())
    
    # Denormalize to output resolution
    fx = fx_n * out_w
    fy = fy_n * out_h
    cx = cx_n * out_w
    cy = cy_n * out_h
    
    K = np.array([
        [fx,  0., cx],
        [0.,  fy, cy],
        [0.,  0., 1.]
    ])
    
    print(f"Loaded K:\n{K}")
    print(f"Input res: {in_w}x{in_h} → Output res: {out_w}x{out_h}")
    print(f"Fisheye omega (distortion): {omega:.4f}")
    print("NOTE: omega != 0, images have fisheye distortion")
    
    return K, omega, (out_w, out_h)


# --- Load image sequence (sorted!) ---
def load_image_sequence(images_folder, out_size=(640, 480)):
    files = sorted(os.listdir(images_folder))
    images = []
    for f in files:
        if f.endswith(('.jpg', '.png')):
            img = cv.imread(os.path.join(images_folder, f), cv.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv.resize(img, out_size)   # resize to 640x480
                images.append(img)
    print(f"Loaded {len(images)} images, shape: {images[0].shape}")
    return images


# --- Quick sanity check ---
if __name__ == "__main__":
    K, omega, out_res = load_tum_mono_calib("dataset/sequence_38/camera.txt")
    images = load_image_sequence("dataset/sequence_38/images/")
    
    # Show first frame to confirm loading works
    cv.imshow("Frame 0", images[0])
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    print(f"Image shape: {images[0].shape}")
    print(f"Expected:    {out_res[1]}x{out_res[0]}")  # h x w
