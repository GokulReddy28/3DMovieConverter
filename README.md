🎥 TriDify – AI-Powered 3D Movie Converter

Convert any 2D video into 3D (Hybrid Depth + Optical Flow + Audio Retention) using advanced AI depth estimation and smart stereo reconstruction.

This project includes:

✅ AI depth estimation (MiDaS DPT-Hybrid)
✅ Motion-aware blended depth
✅ Stereo 3D (Red-Cyan Anaglyph) generation
✅ Audio extraction + re-merge
✅ FPS tracking
✅ ETA prediction
✅ Optimized GUI & CLI
✅ Real-time preview (optional scripts)

🚀 Features
🎞️ Hybrid Depth System

AI depth from MiDaS (DPT-Hybrid)

Optical flow-based motion depth

Smart fusion for stable interpolated frames

🎧 Audio Preservation

Extracts original audio

Merges it back into the converted 3D video

⚡ Performance

FPS tracking

Remaining time estimation

Motion-aware depth smoothing

Optional GPU acceleration (CUDA when available)

🖥️ GUI Application

Simplified GUI for:

Choosing input/output file

Running conversion

Live progress

Preview output

🛠️ CLI Tool

Full control through terminal:

python convert_movie_to_3d_hybrid_audio.py input.mp4 output_3d.mp4

📁 Project Structure
3DProject/
│
├── convert_movie_to_3d_hybrid_audio.py   # Main AI engine
├── tridify_gui.py                        # GUI app
├── real_time_3d_preview.py               # Optional modules
├── real_time_depth.py
├── real_time_anaglyph.py
├── real_time_wiggle_3d.py
│
├── assets/
│   ├── icon.ico
│   ├── logo.png
│   ├── splash.png
│
├── README.md
├── .gitignore

▶️ How to Use (GUI Version)
1️⃣ Launch GUI

Run:

python tridify_gui.py

2️⃣ Select Input & Output

Browse and pick your 2D input video

Choose where to save the 3D output

3️⃣ Start Conversion

You will see:

Progress %

FPS

ETA (Estimated time remaining)

4️⃣ Output

Final 3D video will appear as:

output_3d.mp4

▶️ How to Use (CLI Version)
python convert_movie_to_3d_hybrid_audio.py "input.mp4" "output_3d.mp4"


You will see logs like:

PROG: 85% | FPS: 8.24 | ETA: 00:00:06

🛠️ Build Executable (EXE)

To create a standalone .exe:

Step 1 — Engine EXE
pyinstaller --noconfirm --onefile convert_movie_to_3d_hybrid_audio.py

Step 2 — GUI EXE
pyinstaller --noconfirm --windowed --icon=assets/icon.ico --add-data "assets;assets" tridify_gui.py


Final executables appear in:

dist/

📌 Requirements
Python Dependencies
opencv-python
numpy
torch
transformers
ttk
customtkinter (optional GUI)

External Requirements

FFmpeg installed & added to PATH

GPU (optional) for faster processing

⚙️ How It Works (Simplified)

1️⃣ Extract audio from input
2️⃣ Run AI depth estimation on keyframes
3️⃣ Generate depth using optical flow for in-between frames
4️⃣ Blend depth maps
5️⃣ Convert depth → stereo 3D anaglyph
6️⃣ Re-encode video
7️⃣ Re-merge original audio

📝 License

MIT License – free to use, modify, and distribute.

❤️ Author

Gokul Nanda H V
GitHub: https://github.com/GokulReddy28
