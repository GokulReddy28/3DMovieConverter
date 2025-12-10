# convert_movie_to_3d_hybrid_audio.py
"""
Hybrid converter with audio merge + smooth progress + ETA + FPS + cancel support.

Usage:
    python convert_movie_to_3d_hybrid_audio.py input.mp4 output_3d.mp4

Cancellation:
    - By default, the script checks for a file named "cancel.flag" in the current working directory.
    - You can override the cancel-file path by setting the environment variable:
        TRIDIFY_CANCEL_FILE=C:\path\to\my_cancel_file.flag
    - Create that file (GUI or user) to request a graceful stop. The script will cleanup temporary files.
"""

import sys
import time
import math
import subprocess
import os
import signal
import cv2
import numpy as np
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor

# ------------------ USER TUNABLE ------------------
KEYFRAME_INTERVAL = 30
MAX_SHIFT_BASE = 14
DEPTH_BLEND_ALPHA = 0.6
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
CANCEL_FILE = os.environ.get("TRIDIFY_CANCEL_FILE", "cancel.flag")
# Smoothing factor for FPS EMA (0..1, higher = smoother)
FPS_EMA_ALPHA = 0.15
# --------------------------------------------------

def load_midas():
    print("Loading MiDaS model (DPT-hybrid) on", DEVICE)
    model = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-hybrid-midas",
        use_safetensors=True
    ).to(DEVICE).eval()
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    return model, processor

def compute_midas_depth(model, processor, frame, target_longer=1024):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scale = min(1.0, target_longer / max(h, w))
    if scale < 1.0:
        small = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        small = rgb
    inputs = processor(images=small, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
        depth = out.predicted_depth.squeeze().cpu().numpy()
    depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    depth_resized -= np.min(depth_resized)
    denom = np.max(depth_resized) if np.max(depth_resized) > 1e-6 else 1.0
    depth_resized /= denom
    return depth_resized

def warp_depth_with_flow(prev_depth, flow):
    h, w = prev_depth.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xx - flow[...,0]).astype(np.float32)
    map_y = (yy - flow[...,1]).astype(np.float32)
    warped = cv2.remap(prev_depth.astype(np.float32), map_x, map_y,
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.clip(warped, 0.0, 1.0)

def motion_depth_from_flow(flow):
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    mag = cv2.GaussianBlur(mag, (9,9), 0)
    mag = (mag - mag.min()) / (1e-6 + (mag.max() - mag.min()))
    return mag

def stereo_from_depth(frame, depth_map, max_shift_px):
    H, W = depth_map.shape
    shift_map = (depth_map * max_shift_px).astype(np.int32)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    left_x = np.clip(xx - shift_map, 0, W-1)
    right_x = np.clip(xx + shift_map, 0, W-1)

    left = frame[yy, left_x]
    right = frame[yy, right_x]

    left = cv2.blur(left, (3,3))
    right = cv2.blur(right, (3,3))

    ana = np.zeros_like(left)
    ana[:,:,2] = left[:,:,2]
    ana[:,:,1] = right[:,:,1]
    ana[:,:,0] = right[:,:,0]

    return ana

def readable_time(seconds):
    if seconds < 0 or not math.isfinite(seconds):
        return "--:--:--"
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def check_cancel():
    return os.path.exists(CANCEL_FILE)

def hybrid_convert(input_path, output_path):
    temp_audio = "temp_audio.aac"
    temp_video = "temp_3d_video.mp4"
    start_t = time.time()

    try:
        # Extract audio (quick)
        print("Extracting audio from input video...")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-acodec", "copy", temp_audio
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video: " + input_path)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {w}x{h} @ {fps:.2f} FPS, frames={total}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video, fourcc, fps, (w,h))

        model, processor = load_midas()
        max_shift_px = max(8, int(w * MAX_SHIFT_BASE / 640))

        ret, prev = cap.read()
        if not ret:
            raise RuntimeError("Failed reading first frame")
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        print("Computing AI depth for frame 0")
        ai_depth = compute_midas_depth(model, processor, prev)
        last_ai_depth = ai_depth.copy()
        last_ai_idx = 0

        out.write(stereo_from_depth(prev, ai_depth, max_shift_px))

        frame_idx = 1
        smoothed_fps = None
        last_progress_print = -1

        # We'll use start_t_frame to compute FPS and ETA
        start_t_frame = time.time()

        while True:
            # cancellation check
            if check_cancel():
                print("CANCELLED by cancel file. Cleaning up and exiting.")
                cap.release()
                out.release()
                # remove temps if exist
                try:
                    if os.path.exists(temp_video):
                        os.remove(temp_video)
                except:
                    pass
                try:
                    if os.path.exists(temp_audio):
                        os.remove(temp_audio)
                except:
                    pass
                sys.exit(2)  # non-zero indicates cancelled

            ret, frm = cap.read()
            if not ret:
                break

            t_now = time.time()
            elapsed_from_start = t_now - start_t_frame
            processed_frames = max(1, frame_idx)  # avoid 0 division

            # instantaneous fps and EMA smoothing
            inst_fps = processed_frames / max(1e-6, (t_now - start_t))
            if smoothed_fps is None:
                smoothed_fps = inst_fps
            else:
                smoothed_fps = (FPS_EMA_ALPHA * inst_fps) + ((1.0 - FPS_EMA_ALPHA) * smoothed_fps)

            # Estimate progress & ETA (based on smoothed fps)
            if total > 0:
                frames_done = frame_idx
                progress = int(round((frames_done / total) * 100))
                remaining_frames = max(0, total - frames_done)
                eta_seconds = remaining_frames / max(1e-6, smoothed_fps)
            else:
                # unknown total -> just show processed count
                progress = 0
                eta_seconds = float("nan")

            # Print a compact progress line for GUI parsing:
            # Format: PROG: <int>% | FPS: <float> | ETA: HH:MM:SS
            eta_str = readable_time(eta_seconds)
            print(f"PROG: {progress}% | FPS: {smoothed_fps:.2f} | ETA: {eta_str}")

            # also print a plain percent line (some GUIs expect something like "42%")
            # but only print integer percent if it changed to reduce spam
            if progress != last_progress_print:
                print(f"{progress}%")
                last_progress_print = progress

            # ----- processing for this frame -----
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 3, 15, 2, 5, 1.2, 0)

            warped = warp_depth_with_flow(last_ai_depth, flow)
            motion_depth = motion_depth_from_flow(flow)

            blended_depth = (DEPTH_BLEND_ALPHA * warped) + ((1.0 - DEPTH_BLEND_ALPHA) * motion_depth)
            blended_depth = np.clip(blended_depth, 0.0, 1.0)

            if (frame_idx - last_ai_idx) >= KEYFRAME_INTERVAL:
                print(f"Keyframe at {frame_idx}: computing AI depth")
                ai_depth = compute_midas_depth(model, processor, frm)
                ai_depth = 0.7 * ai_depth + 0.3 * blended_depth
                last_ai_depth = ai_depth.copy()
                last_ai_idx = frame_idx
                use_depth = ai_depth
            else:
                use_depth = blended_depth

            use_depth = np.clip(use_depth ** 0.5, 0.0, 1.0)
            ana = stereo_from_depth(frm, use_depth, max_shift_px)
            out.write(ana)

            prev_gray = gray
            frame_idx += 1
            # -------------------------------------

        cap.release()
        out.release()

        # Final progress 100%
        print("PROG: 100% | FPS: 0.00 | ETA: 00:00:00")
        print("100%")

        # Merge audio + video
        print("Merging 3D video with original audio...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", temp_audio,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # cleanup temporaries
        try:
            if os.path.exists(temp_video):
                os.remove(temp_video)
        except:
            pass
        try:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        except:
            pass

        total_time = time.time() - start_t
        print("Done. Total time:", total_time, "seconds")
        return 0

    except KeyboardInterrupt:
        print("Interrupted by user (KeyboardInterrupt). Cleaning up...")
        try:
            if os.path.exists(temp_video):
                os.remove(temp_video)
        except:
            pass
        try:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        except:
            pass
        return 3

    except SystemExit as se:
        # propagate SystemExit codes
        raise

    except Exception as ex:
        print("ERROR:", str(ex))
        # attempt cleanup
        try:
            if os.path.exists(temp_video):
                os.remove(temp_video)
        except:
            pass
        try:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        except:
            pass
        return 1

# ------------------ CLI ------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_movie_to_3d_hybrid_audio.py input.mp4 output.mp4")
        sys.exit(1)
    rc = hybrid_convert(sys.argv[1], sys.argv[2])
    # exit with returned code so GUI can see non-zero on failure/cancel
    sys.exit(rc if isinstance(rc, int) else 0)
