import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import DPTForDepthEstimation, DPTImageProcessor

# ------------------------------------
# Load MiDaS (DPT-Large) from HuggingFace
# ------------------------------------
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")

# ------------------------------------
# Load image
# ------------------------------------
img = cv2.imread("test.jpg")
if img is None:
    print("❌ Error: test.jpg not found")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ------------------------------------
# Preprocess
# ------------------------------------
inputs = processor(images=img_rgb, return_tensors="pt")

# ------------------------------------
# Predict depth
# ------------------------------------
with torch.no_grad():
    outputs = model(**inputs)
    depth = outputs.predicted_depth.squeeze().numpy()

# Normalize depth for display
depth = (depth - depth.min()) / (depth.max() - depth.min())

# ------------------------------------
# Display Results
# ------------------------------------
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Input Image")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Depth Map (MiDaS DPT-Large)")
plt.imshow(depth, cmap="inferno")
plt.axis("off")

plt.show()
