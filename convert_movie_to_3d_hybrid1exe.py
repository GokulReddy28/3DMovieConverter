# convert_movie_to_3d_hybrid.py
"""
Hybrid 2D->3D converter:
- Uses MiDaS (DPT-hybrid) for keyframe depth (GPU if available)
- Warps depth between keyframes with optical flow (Farneback)
- Blends motion depth for adaptation
- Produces red-cyan anaglyph output
Usage:
    python convert_movie_to_3d_hybrid.py input.mp4 output_3d.mp4
Tune: KEYFRAME_INTERVAL, MAX_SHIFT, DEPTH_BLEND_ALPHA
"""

import sys, time, math
import cv2
import numpy as np
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor

# ------------------ USER TUNABLE ------------------
KEYFRAME_INTERVAL = 30        # compute full AI depth every N frames (30 -> ~1s at 30fps)
MAX_SHIFT_BASE = 14           # base stereo shift strength (px). Scaled by width.
DEPTH_BLEND_ALPHA = 0.6       # blending of AI depth and warped depth (0..1). Higher => trust AI more.
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
# --------------------------------------------------

def load_midas():
    print("Loading MiDaS model (DPT-hybrid) on", DEVICE)
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", use_safetensors=True).to(DEVICE).eval()
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    return model, processor

def compute_midas_depth(model, processor, frame, target_longer=1024):
    # frame: BGR uint8 HxW
    # Resize while preserving aspect ratio, target size for faster inference
    h, w = frame.shape[:2]
    # MiDaS expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Optionally resize to keep longest side = target_longer for speed
    scale = min(1.0, target_longer / max(h, w))
    if scale < 1.0:
        small = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        small = rgb
    inputs = processor(images=small, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
        depth = out.predicted_depth.squeeze().cpu().numpy()
    # Resize depth back to original frame size
    depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    # Normalize depth to 0..1
    depth_resized = depth_resized - np.min(depth_resized)
    denom = np.max(depth_resized) if np.max(depth_resized) > 1e-6 else 1.0
    depth_resized = depth_resized / denom
    return depth_resized

def warp_depth_with_flow(prev_depth, flow):
    """
    Warp prev_depth forward using optical flow from prev->cur.
    flow is the (dx,dy) field where flow[y,x] = (u,v) mapping prev->cur
    We want depth_cur_est(x,y) = prev_depth(x - u, y - v)
    """
    h, w = prev_depth.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xx - flow[...,0]).astype(np.float32)
    map_y = (yy - flow[...,1]).astype(np.float32)
    warped = cv2.remap(prev_depth.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # clamp/normalize
    warped = np.clip(warped, 0.0, 1.0)
    return warped

def motion_depth_from_flow(flow):
    # Compute a normalized motion magnitude depth proxy (0..1)
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    # blur & normalize
    mag = cv2.GaussianBlur(mag, (9,9), 0)
    mag = (mag - mag.min()) / (1e-6 + (mag.max() - mag.min()))
    return mag

def stereo_from_depth(frame, depth_map, max_shift_px):
    """Vectorized stereo generation: left/right by horizontal shifts"""
    H, W = depth_map.shape
    shift_map = (depth_map * max_shift_px).astype(np.int32)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    left_x = np.clip(xx - shift_map, 0, W-1)
    right_x = np.clip(xx + shift_map, 0, W-1)
    left = frame[yy, left_x]
    right = frame[yy, right_x]
    # small blur to smooth seams
    left = cv2.blur(left, (3,3))
    right = cv2.blur(right, (3,3))
    # create anaglyph (red from left, green/blue from right)
    ana = np.zeros_like(left)
    ana[:,:,2] = left[:,:,2]   # R
    ana[:,:,1] = right[:,:,1]  # G
    ana[:,:,0] = right[:,:,0]  # B
    return ana

def hybrid_convert(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + input_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {w}x{h} @ {fps:.2f} FPS, frames={total}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w,h))

    # Model load
    model, processor = load_midas()

    # dynamic max shift scale by width
    max_shift_px = max(8, int(w * MAX_SHIFT_BASE / 640))  # scale from base at 640w

    # read first frame
    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Failed reading first frame")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # compute AI depth for first frame
    print("Computing AI depth for frame 0")
    ai_depth = compute_midas_depth(model, processor, prev)
    last_ai_depth = ai_depth.copy()
    last_ai_frame = prev.copy()
    last_ai_idx = 0

    # write first output (neutral or small shift)
    out.write(stereo_from_depth(prev, ai_depth, max_shift_px))

    frame_idx = 1
    start_t = time.time()
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        # compute flow prev -> cur
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 2, 5, 1.2, 0)

        # Warp last AI depth forward using flow
        warped = warp_depth_with_flow(last_ai_depth, flow)

        # Compute short-term motion depth from this flow
        motion_depth = motion_depth_from_flow(flow)

        # blend warped AI depth with motion depth to adapt to new motion (and reduce ghosting)
        # alpha controls trust in AI vs motion; also scale motion to 0..1
        blended_depth = (DEPTH_BLEND_ALPHA * warped) + ((1.0 - DEPTH_BLEND_ALPHA) * motion_depth)
        # optional re-normalization
        blended_depth = np.clip(blended_depth, 0.0, 1.0)

        # If this frame is a keyframe -> compute fresh AI depth
        if (frame_idx - last_ai_idx) >= KEYFRAME_INTERVAL:
            print(f"Keyframe at {frame_idx}: computing AI depth")
            ai_depth = compute_midas_depth(model, processor, frm)
            # optionally blend AI depth & blended_depth (helps stability)
            ai_depth = 0.7 * ai_depth + 0.3 * blended_depth
            last_ai_depth = ai_depth.copy()
            last_ai_frame = frm.copy()
            last_ai_idx = frame_idx
            use_depth = ai_depth
        else:
            # use blended depth
            use_depth = blended_depth

        # optionally apply non-linear boost for stronger pop-out (tune power <1 => stronger)
        boost_power = 0.5   # lower -> stronger pop
        use_depth = np.clip(use_depth ** boost_power, 0.0, 1.0)

        # generate stereo/anaglyph
        ana = stereo_from_depth(frm, use_depth, max_shift_px)
        out.write(ana)

        # update
        prev_gray = gray
        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - start_t
            fps_proc = frame_idx / max(1.0, elapsed)
            print(f"frame {frame_idx}/{total} — proc fps ~ {fps_proc:.1f}")

    cap.release()
    out.release()
    total_time = time.time() - start_t
    print("Done. Time:", total_time, "s — avg fps:", (frame_idx / total_time) if total_time>0 else 0)

# ------------------ CLI ------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_movie_to_3d_hybrid.py input.mp4 output.mp4")
        sys.exit(1)
    inp = sys.argv[1]
    outp = sys.argv[2]
    hybrid_convert(inp, outp)
