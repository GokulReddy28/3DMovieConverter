import cv2
import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

import sys

input_video = sys.argv[1]
output_video = sys.argv[2]

print("Using:", input_video)

# ---------------------------
# LOAD MODEL USING SAFETENSORS ONLY
# ---------------------------
print("Loading MiDaS Hybrid model on GPU...")

model = DPTForDepthEstimation.from_pretrained(
    "Intel/dpt-hybrid-midas",
    use_safetensors=True,   # <-- IMPORTANT FIX
    torch_dtype=torch.float32
).to("cuda").eval()

processor = DPTImageProcessor.from_pretrained(
    "Intel/dpt-hybrid-midas"
)

# ---------------------------
# OPEN VIDEO
# ---------------------------
cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video Loaded: {w} x {h} @ {fps} FPS")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

# ---------------------------
# FAST 3D GENERATION
# ---------------------------
def generate_3d(frame, depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    shift_map = (depth * 12).astype(np.int32)

    H, W = frame.shape[:2]
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    left = frame[yy, np.clip(xx - shift_map, 0, W - 1)]
    right = frame[yy, np.clip(xx + shift_map, 0, W - 1)]

    anaglyph = np.zeros_like(frame)
    anaglyph[:, :, 2] = left[:, :, 2]
    anaglyph[:, :, 1] = right[:, :, 1]
    anaglyph[:, :, 0] = right[:, :, 0]

    return anaglyph

# ---------------------------
# PROCESS FRAMES
# ---------------------------
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (480, 270))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    inputs = processor(images=rgb, return_tensors="pt").to("cuda")

    with torch.no_grad():
        depth = model(**inputs).predicted_depth.squeeze().cpu().numpy()

    out.write(generate_3d(frame, depth))

    frame_id += 1
    if frame_id % 30 == 0:
        print("Processed frame:", frame_id)

cap.release()
out.release()

print("\nDONE! Saved:", output_video)
