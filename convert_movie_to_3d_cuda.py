import cv2
import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

# ----------------------------
# Load MiDaS AI Model on GPU
# ----------------------------
print("Loading depth model on GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
# processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid").to("cuda").eval()
processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid")

# ----------------------------
# Input / Output Paths
# ----------------------------
input_video = "test.mp4"
output_video = "output_3d_movie.mp4"

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if w == 0 or h == 0 or fps == 0:
    raise Exception("ERROR: Cannot read video! Check filename or path.")

print(f"Video Loaded: {w} x {h} @ {fps} FPS")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

# ----------------------------
# Stereo 3D Generation
# ----------------------------
def generate_stereo(frame, depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    # Compute shift map (vectorized, no loops!)
    shift_map = (depth * 12).astype(np.int32)

    H, W = frame.shape[:2]

    # Create meshgrid
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    # LEFT image = shift left
    left_x = np.clip(xx - shift_map, 0, W - 1)
    left = frame[yy, left_x]

    # RIGHT image = shift right
    right_x = np.clip(xx + shift_map, 0, W - 1)
    right = frame[yy, right_x]

    # Simple anaglyph
    anaglyph = np.zeros_like(frame)
    anaglyph[:, :, 2] = left[:, :, 2]   # R
    anaglyph[:, :, 1] = right[:, :, 1]  # G
    anaglyph[:, :, 0] = right[:, :, 0]  # B

    return anaglyph


# ----------------------------
# Process Video with GPU
# ----------------------------
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)   # <-- MOVED TO GPU

    with torch.no_grad():
        depth_map = model(**inputs).predicted_depth.squeeze().cpu().numpy()

    anaglyph = generate_stereo(frame, depth_map)
    out.write(anaglyph)

    count += 1
    print(f"Processed frame: {count}", end="\r")

cap.release()
out.release()

print("\nDONE! Saved as:", output_video)
