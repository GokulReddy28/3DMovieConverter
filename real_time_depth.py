import cv2
import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

# Load model
print("Loading AI model...")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")

# Open webcam
cap = cv2.VideoCapture(0)

print("Starting webcam... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    inputs = processor(images=rgb, return_tensors="pt")

    with torch.no_grad():
        depth = model(**inputs).predicted_depth.squeeze().numpy()

    # Normalize depth for display
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    # Resize depth map to webcam shape
    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    # Convert to heatmap
    depth_colored = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    # Show both windows
    cv2.imshow("Webcam - Input", frame)
    cv2.imshow("Depth Map - Real Time", depth_colored)

    # Quit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
