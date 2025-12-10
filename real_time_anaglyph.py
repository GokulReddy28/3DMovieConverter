import cv2
import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

# ----------------------------
# Load model
# ----------------------------
print("Loading MiDaS model...")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")

cap = cv2.VideoCapture(0)
print("Starting Anaglyph 3D... Press Q to quit.")

# ----------------------------------------
# Function: Create Stereo Left/Right Views
# ----------------------------------------
def generate_stereo(frame, depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    max_shift = 15

    h, w = frame.shape[:2]
    left = np.zeros_like(frame)
    right = np.zeros_like(frame)

    for y in range(h):
        for x in range(w):
            shift = int(depth[y, x] * max_shift)
            if x - shift >= 0:
                left[y, x - shift] = frame[y, x]
            if x + shift < w:
                right[y, x + shift] = frame[y, x]

    # Fill gaps
    left = cv2.inpaint(left, (left == 0).all(axis=2).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    right = cv2.inpaint(right, (right == 0).all(axis=2).astype(np.uint8), 3, cv2.INPAINT_TELEA)

    return left, right

# ----------------------------------------
# Real-time Loop
# ----------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")

    # Predict depth
    with torch.no_grad():
        depth = model(**inputs).predicted_depth.squeeze().numpy()

    left, right = generate_stereo(frame, depth)

    # -----------------------
    # ANAGLYPH 3D CREATION
    # -----------------------
    anaglyph = np.zeros_like(frame)
    anaglyph[:,:,2] = left[:,:,2]     # Red channel  
    anaglyph[:,:,0] = right[:,:,0]    # Blue channel
    anaglyph[:,:,1] = right[:,:,1]    # Green channel

    cv2.imshow("Anaglyph 3D (Red–Cyan)", anaglyph)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
