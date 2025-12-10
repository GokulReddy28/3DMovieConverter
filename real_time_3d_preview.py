import cv2
import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

# Load model
print("Loading AI model...")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")

cap = cv2.VideoCapture(0)

print("Starting 3D Preview... Press Q to quit.")

def generate_stereo(frame, depth):
    # Normalize depth
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    # Smooth depth to remove artifacts
    depth = cv2.bilateralFilter(depth.astype(np.float32), 9, 75, 75)

    max_shift = 15  # Adjust 3D strength

    left = np.zeros_like(frame)
    right = np.zeros_like(frame)

    h, w = frame.shape[:2]

    for y in range(h):
        for x in range(w):
            shift = int(depth[y, x] * max_shift)

            # LEFT eye (shift left)
            new_x_left = x - shift
            if 0 <= new_x_left < w:
                left[y, new_x_left] = frame[y, x]

            # RIGHT eye (shift right)
            new_x_right = x + shift
            if 0 <= new_x_right < w:
                right[y, new_x_right] = frame[y, x]

    # Fill holes (interpolation)
    left = cv2.inpaint(left, (left == 0).all(axis=2).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    right = cv2.inpaint(right, (right == 0).all(axis=2).astype(np.uint8), 3, cv2.INPAINT_TELEA)

    return left, right


while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")

    with torch.no_grad():
        depth = model(**inputs).predicted_depth.squeeze().numpy()

    left, right = generate_stereo(frame, depth)

    # SIDE-BY-SIDE 3D VIEW
    sbs = np.hstack((left, right))

    cv2.imshow("3D SBS Preview (Wear Red-Cyan or VR)", sbs)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
