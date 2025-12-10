import cv2
import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

# ----------------------------
# Load MiDaS AI model ON GPU
# ----------------------------
print("Loading depth model on GPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")

# ----------------------------
# Input / Output Paths
# ----------------------------
input_video = "mybigmovie.mp4"
output_video = "output_3d_movie.mp4"

# ----------------------------
# Open input movie
# ----------------------------
cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if w == 0 or h == 0 or fps == 0:
    raise Exception("ERROR: Cannot read video! Check filename/path.")

print(f"Video Loaded: {w}x{h} @ {fps} FPS")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

# ----------------------------
# Stereo + Anaglyph 3D
# ----------------------------
def generate_stereo(frame, depth_map):
    depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    max_shift = 12
    H, W = frame.shape[:2]

    left  = np.zeros_like(frame)
    right = np.zeros_like(frame)

    for y in range(H):
        for x in range(W):
            shift = int(depth[y, x] * max_shift)

            if x - shift >= 0:
                left[y, x - shift] = frame[y, x]
            if x + shift < W:
                right[y, x + shift] = frame[y, x]

    mask_left = (left == 0).all(axis=2).astype(np.uint8)
    left = cv2.inpaint(left, mask_left, 3, cv2.INPAINT_TELEA)

    mask_right = (right == 0).all(axis=2).astype(np.uint8)
    right = cv2.inpaint(right, mask_right, 3, cv2.INPAINT_TELEA)

    anaglyph = np.zeros_like(frame)
    anaglyph[:, :, 2] = left[:, :, 2]
    anaglyph[:, :, 1] = right[:, :, 1]
    anaglyph[:, :, 0] = right[:, :, 0]

    return anaglyph

# ----------------------------
# Process video
# ----------------------------
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        depth_map = model(**inputs).predicted_depth.squeeze().cpu().numpy()

    anaglyph = generate_stereo(frame, depth_map)
    out.write(anaglyph)

    frame_count += 1
    print("Processed frame:", frame_count, end="\r")

cap.release()
out.release()

print("\nDONE! Saved as:", output_video)
