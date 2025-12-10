# 🎬 TriDify – AI-Powered 3D Movie Converter  
Convert any 2D video into immersive **3D (Hybrid Depth + Motion)** using AI depth estimation, optical flow, and audio-preserving rendering.

---

## 🚀 Features

✔ **AI Depth Estimation** using MiDaS DPT-Hybrid  
✔ **Hybrid Depth + Motion Flow Algorithm**  
✔ **Smooth Stereo (Left–Right) Rendering**  
✔ **Original Audio Preservation**  
✔ **Fast Optical Flow & Efficient Warping**  
✔ **Progress Tracker (FPS, ETA, %) when using EXE**  
✔ **GPU Acceleration (CUDA if available)**  
✔ **Supports any video format (MP4, MKV, etc.)**  
✔ **Optional GUI Application**  
✔ **Offline-ready engine (no internet needed)**  

---

## 🧠 How It Works

### 1️⃣ Depth Estimation  
MiDaS DPT-Hybrid predicts a depth map for keyframes.

### 2️⃣ Motion-Aware Depth  
Optical flow propagates depth forward between frames for speed.

### 3️⃣ Hybrid Depth Blending  
Depth = 70% AI depth + 30% motion depth  
→ Produces smoother & more stable 3D.

### 4️⃣ Stereo Generation  
Each pixel is shifted left/right based on depth → anaglyph 3D output.

### 5️⃣ Audio Merge  
FFmpeg merges original audio back with the generated 3D video.

---

## 📂 Project Structure
3DMovieConverter/
│── convert_movie_to_3d_hybrid_audio.py # Main engine (CLI)
│── tridify_gui.py # GUI version (optional)
│── assets/ # Icons & images
│── test scripts/ # GPU tests, real-time tools
│── README.md # This file
│── .gitignore


---

## 🛠️ Installation

### **1. Install Python 3.10–3.12**
https://www.python.org/downloads/

### **2. Install required libraries**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python transformers numpy


If CPU only:

pip install torch opencv-python transformers numpy

▶️ Run Conversion (CLI Mode)
python convert_movie_to_3d_hybrid_audio.py input.mp4 output_3d.mp4


Example:

python convert_movie_to_3d_hybrid_audio.py myvideo.mp4 myvideo_3d.mp4

🖥️ Run GUI Version
python tridify_gui.py


You can browse:

Input video

Output file

Engine EXE

Start conversion

Preview output

📦 Build Standalone EXE (Windows)
First install PyInstaller:
pip install pyinstaller

Build engine EXE:
pyinstaller --noconfirm --onefile convert_movie_to_3d_hybrid_audio.py

Build GUI EXE:
pyinstaller --noconfirm --windowed --icon=assets/icon.ico --add-data "assets;assets" tridify_gui.py


Your EXEs will appear inside:

dist/

📊 Performance
Hardware	FPS	Notes
GTX 1650	~8 FPS	Smooth conversion
RTX 3050	~14 FPS	Fast 3D conversion
CPU Only	1–2 FPS	Very slow

GPU recommended.

🎨 Preview of 3D Output (Anaglyph)

Red = left eye

Cyan = right eye

Works with any red/cyan 3D glasses

📝 Known Limitations

⚠ Slow on CPU
⚠ Not real-time for large videos
⚠ Anaglyph colors may slightly distort original colors

🤝 Contributing

Pull requests welcome!
If you improve depth/blending/GUI, feel free to contribute.

⭐ Support the Project

If this helped you:

⭐ Star the repo
🍴 Fork it
🐛 Report issues

📬 Contact

Author: GokulNanda HV (Gokul Reddy)
GitHub: https://github.com/GokulReddy28
