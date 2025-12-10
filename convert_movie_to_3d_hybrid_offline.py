"""
OFFLINE HYBRID 3D Converter (NO HUGGINGFACE)
--------------------------------------------
- Uses MiDaS v2_small_256 (Torch Hub) — fast, reliable, offline
- Keyframe AI depth + optical flow + blended depth
- Creates red–cyan 3D anaglyph video
- Fully GPU accelerated (if CUDA available)
Usage:
    python convert_movie_to_3d_hybrid_offline.py input.mp4 output.mp4
"""

import sys, time
import cv2
import numpy as np
import torch
from tqdm import tqdm

# ----------------- SETTINGS -----------------
KEYFRAME_INTERVAL = 30
DEPTH_BLEND_ALPHA = 0.6
BOOST_POWER = 0.55
MAX_SHIFT_BASE = 14

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ----------------- LOAD MIDAS -----------------
def load_midas():
    print("\nLoading MiDaS v2_small_256 (TorchHub)...")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.to(DEVICE).eval()

    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return model, transform

# ----------------- AI DEPTH -----------------
def compute_depth(model, transform, frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(DEVICE)

    with torch.no_grad():
        prediction = model(input_batch)

    depth = prediction.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    return cv2.resize(depth, (frame.shape[1], frame.shape[0]))

# ----------------- OPTICAL FLOW WARP -----------------
def warp_depth(prev_depth, flow):
    h, w = prev_depth.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xx - flow[..., 0]).astype(np.float32)
    map_y = (yy - flow[..., 1]).astype(np.float32)

    warped = cv2.remap(prev_depth.astype(np.float32), map_x, map_y,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.clip(warped, 0, 1)

def motion_from_flow(flow):
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    mag = cv2.GaussianBlur(mag, (9, 9), 0)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    return mag

# ----------------- CREATE ANAGLYPH -----------------
def stereo_from_depth(frame, depth, max_shift):
    h, w = depth.shape
    shiftmap = (depth * max_shift).astype(np.int32)

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    left_x = np.clip(xx - shiftmap, 0, w - 1)
    right_x = np.clip(xx + shiftmap, 0, w - 1)

    left = frame[yy, left_x]
    right = frame[yy, right_x]

    ana = np.zeros_like(frame)
    ana[:, :, 2] = left[:, :, 2]
    ana[:, :, 1] = right[:, :, 1]
    ana[:, :, 0] = right[:, :, 0]
    return ana

# ----------------- MAIN CONVERSION -----------------
def convert_video(input_path, output_path):

    model, transform = load_midas()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open: " + input_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_shift = int(w * MAX_SHIFT_BASE / 640)

    print(f"\nVideo {w}x{h}, {fps}fps, frames={total}\n")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ret, prev = cap.read()
    if not ret: raise RuntimeError("Failed to read first frame")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # FIRST FRAME AI DEPTH
    print("Computing AI depth for frame 0...")
    ai_depth = compute_depth(model, transform, prev)
    last_ai = ai_depth.copy()
    last_ai_idx = 0

    out.write(stereo_from_depth(prev, ai_depth, max_shift))

    pbar = tqdm(total=total)

    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.3, 0)

        warped = warp_depth(last_ai, flow)
        motion = motion_from_flow(flow)

        blended = DEPTH_BLEND_ALPHA * warped + (1 - DEPTH_BLEND_ALPHA) * motion

        # Keyframe AI depth refresh
        if idx - last_ai_idx >= KEYFRAME_INTERVAL:
            print(f"Keyframe {idx}: AI depth...")
            fresh = compute_depth(model, transform, frame)
            blended = 0.7 * fresh + 0.3 * blended
            last_ai = blended.copy()
            last_ai_idx = idx

        # Boost depth
        depth_final = np.clip(blended ** BOOST_POWER, 0, 1)

        anaglyph = stereo_from_depth(frame, depth_final, max_shift)
        out.write(anaglyph)

        prev_gray = gray
        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print("\n✔ DONE! Saved:", output_path)

# ----------------- CLI -----------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_movie_to_3d_hybrid_offline.py input.mp4 output.mp4")
        sys.exit(0)

    convert_video(sys.argv[1], sys.argv[2])
