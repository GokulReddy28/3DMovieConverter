# tri_dify_desktop.py
import sys, os, subprocess, time, shlex
from pathlib import Path
from threading import Thread
import psutil

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QSlider, QComboBox
)

# ---------------- CONFIG ----------------
CONVERTER_SCRIPT = "convert_movie_to_3d_cuda.py"   # must accept: python convert_movie_to_3d_cuda.py <input> <output>
ASSETS_DIR = Path("assets")
LOGO_PATH = ASSETS_DIR / "logo.png"
SPLASH_PATH = ASSETS_DIR / "splash.png"
ICON_PATH = ASSETS_DIR / "icon.ico"
# ----------------------------------------

def resource_path(p: Path):
    return str(p.resolve()) if p.exists() else None

# ---------------- Worker (runs converter) ----------------
class ConverterWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  # frames_processed, total_frames(optional 0)
    finished_signal = pyqtSignal(int)       # return code

    def __init__(self, input_path, output_path, parent=None):
        super().__init__(parent)
        self.input_path = str(Path(input_path).resolve())
        self.output_path = str(Path(output_path).resolve())
        self.proc = None
        self._suspended = False

    def run(self):
        # Build command (use same Python interpreter)
        script = Path(CONVERTER_SCRIPT).resolve()
        if not script.exists():
            self.log_signal.emit(f"[ERROR] Converter not found: {script}")
            self.finished_signal.emit(1)
            return

        cmd = f'"{sys.executable}" "{script}" "{self.input_path}" "{self.output_path}"'
        self.log_signal.emit(f"[INFO] Starting conversion:\n{cmd}")
        try:
            self.proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Failed to start: {e}")
            self.finished_signal.emit(1)
            return

        # Read stdout line-by-line
        frames = 0
        total_frames = 0
        try:
            for rawline in self.proc.stdout:
                line = rawline.rstrip("\n")
                self.log_signal.emit(line)
                # Example parse: "Processed frame: 123" or "Video Loaded: 1920 x 800 @ 23.97 FPS"
                if "Processed frame" in line:
                    try:
                        # try to extract integer
                        frames = int(line.split()[-1])
                        self.progress_signal.emit(frames, total_frames)
                    except:
                        pass
                if "Video Loaded:" in line and "@" in line:
                    # try to parse fps width height for ETA later (not exact)
                    parts = line.split()
                    try:
                        w = int(parts[2])
                        h = int(parts[4])
                        fps_token = parts[-2]
                        fps = float(parts[-1]) if parts[-1].replace('.','',1).isdigit() else 0.0
                        # Not calculating total frames here because it requires duration; keep 0
                        total_frames = 0
                    except:
                        pass
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Streaming error: {e}")

        rc = self.proc.wait() if self.proc else 1
        self.log_signal.emit(f"[INFO] Converter exited with code {rc}")
        self.finished_signal.emit(rc)

    def pause(self):
        if self.proc and psutil.pid_exists(self.proc.pid):
            p = psutil.Process(self.proc.pid)
            p.suspend()
            self._suspended = True
            self.log_signal.emit("[INFO] Converter suspended")

    def resume(self):
        if self.proc and psutil.pid_exists(self.proc.pid):
            p = psutil.Process(self.proc.pid)
            p.resume()
            self._suspended = False
            self.log_signal.emit("[INFO] Converter resumed")

    def cancel(self):
        if self.proc and psutil.pid_exists(self.proc.pid):
            p = psutil.Process(self.proc.pid)
            p.terminate()
            self.log_signal.emit("[INFO] Converter terminated")

# ---------------- Main UI ----------------
class TriDifyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TriDify — 2D → 3D Movie Converter")
        icon_file = resource_path(ICON_PATH)
        if icon_file:
            self.setWindowIcon(QIcon(icon_file))

        self.resize(1100, 700)
        self.setStyleSheet(self._style_sheet())

        # Top: logo + title (if logo exists)
        self.logo_label = QLabel()
        logo_file = resource_path(LOGO_PATH)
        if logo_file:
            pix = QPixmap(logo_file).scaledToHeight(64, Qt.TransformationMode.SmoothTransformation)
            self.logo_label.setPixmap(pix)
        else:
            self.logo_label.setText("TriDify")
            self.logo_label.setStyleSheet("font-size:24px; font-weight:700;")

        # Input / Output lines
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Drop or browse large video file (mp4/mkv/avi)")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_input)

        self.output_line = QLineEdit(str(Path.cwd() / "output_3d_movie.mp4"))
        self.output_line.setPlaceholderText("Output file path")
        self.out_browse = QPushButton("Save As")
        self.out_browse.clicked.connect(self.browse_output)

        # Options
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Anaglyph (Red-Cyan)", "Side-by-Side (SBS)", "Wiggle / Preview"])
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 20)
        self.strength_slider.setValue(10)

        # GPU status
        self.gpu_label = QLabel(self.get_gpu_text())

        # Preview placeholders
        self.left_preview = QLabel("Input Preview")
        self.left_preview.setFixedSize(420, 240)
        self.left_preview.setStyleSheet("background:#111; border-radius:6px;")

        self.right_preview = QLabel("3D Preview")
        self.right_preview.setFixedSize(420, 240)
        self.right_preview.setStyleSheet("background:#111; border-radius:6px;")

        # Buttons
        self.start_btn = QPushButton("Start Conversion")
        self.start_btn.clicked.connect(self.start_conversion)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_conversion)
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.setEnabled(False)
        self.resume_btn.clicked.connect(self.resume_conversion)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_conversion)

        # Logs / progress
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.progress_label = QLabel("Frames: 0")
        self.eta_label = QLabel("ETA: —")

        # Layout assembly
        top_h = QHBoxLayout()
        top_h.addWidget(self.logo_label)
        top_h.addStretch()
        top_h.addWidget(self.gpu_label)

        path_h = QHBoxLayout()
        path_h.addWidget(self.input_line)
        path_h.addWidget(self.browse_btn)

        out_h = QHBoxLayout()
        out_h.addWidget(self.output_line)
        out_h.addWidget(self.out_browse)

        opts_h = QHBoxLayout()
        opts_h.addWidget(QLabel("Mode:"))
        opts_h.addWidget(self.mode_combo)
        opts_h.addWidget(QLabel("Depth Strength:"))
        opts_h.addWidget(self.strength_slider)
        opts_h.addStretch()
        opts_h.addWidget(self.progress_label)
        opts_h.addWidget(self.eta_label)

        preview_h = QHBoxLayout()
        preview_h.addWidget(self.left_preview)
        preview_h.addWidget(self.right_preview)

        btn_h = QHBoxLayout()
        btn_h.addWidget(self.start_btn)
        btn_h.addWidget(self.pause_btn)
        btn_h.addWidget(self.resume_btn)
        btn_h.addWidget(self.cancel_btn)

        main_l = QVBoxLayout()
        main_l.addLayout(top_h)
        main_l.addLayout(path_h)
        main_l.addLayout(out_h)
        main_l.addLayout(opts_h)
        main_l.addLayout(preview_h)
        main_l.addLayout(btn_h)
        main_l.addWidget(QLabel("Log:"))
        main_l.addWidget(self.log)

        self.setLayout(main_l)

        # Worker reference
        self.worker = None
        self._started_time = None
        self._frames_processed = 0

        # Drag-drop
        self.setAcceptDrops(True)

    def _style_sheet(self):
        return """
        QWidget{ background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #0d0d0d, stop:1 #111111); color:#ddd; font-family:Inter, Poppins; }
        QPushButton{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7a34ff, stop:1 #00d4ff); color:#fff; padding:8px 14px; border-radius:8px; font-weight:600;}
        QPushButton:disabled{ opacity:0.5; }
        QLineEdit{ background:#121212; border-radius:6px; padding:8px; color:#ddd; }
        QComboBox{ background:#121212; border-radius:6px; padding:6px; color:#ddd; }
        QLabel{ color:#ddd; }
        QTextEdit{ background:#0b0b0b; border:1px solid #222; color:#ddd; border-radius:6px; }
        QSlider::groove:horizontal { height:8px; background:#222; border-radius:4px; }
        QSlider::handle:horizontal { background: #8A2BE2; width:14px; border-radius:7px; }
        """

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.input_line.setText(urls[0].toLocalFile())
            base = Path(urls[0].toLocalFile()).stem
            self.output_line.setText(str(Path.cwd() / f"{base}_3d.mp4"))

    def get_gpu_text(self):
        try:
            import torch
            if torch.cuda.is_available():
                return f"GPU detected: {torch.cuda.get_device_name(0)} (CUDA)"
            else:
                return "GPU not detected (using CPU)"
        except Exception:
            return "Torch not available"

    def browse_input(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Video", str(Path.home()), "Video Files (*.mp4 *.mkv *.avi *.mov)")
        if f:
            self.input_line.setText(f)
            base = Path(f).stem
            self.output_line.setText(str(Path.cwd() / f"{base}_3d.mp4"))

    def browse_output(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save Output As", str(Path.cwd() / "output_3d_movie.mp4"), "MP4 Video (*.mp4)")
        if f:
            self.output_line.setText(f)

    def start_conversion(self):
        inp = self.input_line.text().strip()
        outp = self.output_line.text().strip()
        if not inp or not Path(inp).exists():
            self.log.append("[ERROR] Input file missing or invalid.")
            return
        if not outp:
            self.log.append("[ERROR] Output path invalid.")
            return

        # disable UI controls
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)

        # start worker
        self.worker = ConverterWorker(inp, outp)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self._started_time = time.time()
        self.worker.start()
        self.log.append("[INFO] Conversion started in background.")

    def pause_conversion(self):
        if self.worker:
            self.worker.pause()
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(True)

    def resume_conversion(self):
        if self.worker:
            self.worker.resume()
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)

    def cancel_conversion(self):
        if self.worker:
            self.worker.cancel()
            self.log.append("[INFO] Cancel requested.")

    def append_log(self, text):
        self.log.append(text)
        # keep last lines visible
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def on_progress(self, frames, total):
        self._frames_processed = frames
        self.progress_label.setText(f"Frames: {frames}")
        # estimate ETA using elapsed time and fps approximate
        if self._started_time and frames > 2:
            elapsed = time.time() - self._started_time
            rate = frames / elapsed
            # if total known compute eta, else show approx per-frame rate
            if total and total > 0:
                remaining = max(0, total - frames)
                eta = remaining / rate
                self.eta_label.setText(f"ETA: {int(eta)}s")
            else:
                self.eta_label.setText(f"Rate: {rate:.1f} fps")

    def on_finished(self, code):
        self.append_log(f"[INFO] Conversion finished (code {code})")
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText("Frames: 0")
        self.eta_label.setText("ETA: —")

# ---------------- Splash (non-blocking) ----------------
def show_splash(parent=None):
    splash_file = resource_path(SPLASH_PATH)
    if not splash_file:
        return None
    splash = QLabel()
    pix = QPixmap(splash_file)
    splash.setPixmap(pix.scaled(900, 506, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    splash.setWindowFlag(Qt.WindowType.FramelessWindowHint)
    splash.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    splash.show()
    QTimer.singleShot(1800, splash.close)  # show for 1.8 seconds
    return splash

# ---------------- Main ----------------
def main():
    app = QApplication(sys.argv)
    # show splash
    splash = show_splash()
    w = TriDifyApp()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
