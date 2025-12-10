# app.py
import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

PROJECT_DIR = Path.cwd()
ASSETS_DIR = PROJECT_DIR / "assets"
ICON_PATH = str(ASSETS_DIR / "icon.ico")
SPLASH_PATH = str(ASSETS_DIR / "splash.png")
LOGO_PATH = str(ASSETS_DIR / "logo.png")

APP_TITLE = "TriDify - 3D Movie Converter (Optical Flow)"

# -------------------------
# Worker thread for conversion
# -------------------------
class ConverterWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)          # percent
    status = QtCore.pyqtSignal(str)            # text
    finished = QtCore.pyqtSignal(str)          # output_path
    error = QtCore.pyqtSignal(str)

    def __init__(self, input_path: str, output_path: str, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.output_path = output_path
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.status.emit("Opening video...")
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                self.error.emit("Unable to open video.")
                return

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.status.emit(f"Video: {w}x{h} @ {int(fps)} FPS — {total} frames")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

            # Parameters
            max_shift_px = max(8, w // 80)   # 3D strength, scaled with width
            pyr_scale = 0.5
            levels = 3
            winsize = 15
            iterations = 2
            poly_n = 5
            poly_sigma = 1.2

            # Read first frame
            ret, prev_frame = cap.read()
            if not ret:
                self.error.emit("Could not read first frame.")
                cap.release()
                out.release()
                return

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_idx = 0

            # Write first frame as neutral (no shift)
            left = prev_frame.copy()
            right = prev_frame.copy()
            anaglyph = self._create_anaglyph(left, right)
            out.write(anaglyph)
            frame_idx += 1
            if total:
                self.progress.emit(int(frame_idx * 100 / total))

            # Processing loop
            while True:
                if self._stop:
                    self.status.emit("Stopped by user.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Optical flow from previous -> current
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                                    None,
                                                    pyr_scale, levels, winsize,
                                                    iterations, poly_n, poly_sigma, 0)

                # magnitude of flow (motion strength) -> depth proxy
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

                # Smooth and normalize magnitude
                mag = cv2.GaussianBlur(mag, (7, 7), 0)
                # downscale-normalize helps stability
                mag_small = cv2.resize(mag, (w // 4, h // 4))
                mag_small = (mag_small - mag_small.min()) / (1e-6 + (mag_small.max() - mag_small.min()))
                mag_norm = cv2.resize(mag_small, (w, h), interpolation=cv2.INTER_LINEAR)

                # generate left/right with vectorized shifts
                left_frame, right_frame = self._stereo_from_depth(frame, mag_norm, max_shift_px)

                anaglyph = self._create_anaglyph(left_frame, right_frame)
                out.write(anaglyph)

                prev_gray = gray
                frame_idx += 1

                if total:
                    pct = int(frame_idx * 100 / total)
                    self.progress.emit(pct)
                else:
                    # if total unknown, just emit incremental updates
                    self.progress.emit(min(99, frame_idx % 100))

                # occasional status update
                if frame_idx % 50 == 0:
                    self.status.emit(f"Processed frame: {frame_idx}")

            cap.release()
            out.release()
            self.progress.emit(100)
            self.status.emit("Conversion finished.")
            self.finished.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e))

    def _stereo_from_depth(self, frame, depth_map, max_shift):
        """
        depth_map: float 0..1, same WxH as frame
        returns left and right frames (H,W,3)
        Vectorized shift using numpy indexing (fast).
        """
        H, W = depth_map.shape
        # compute integer shift per pixel
        shift_map = (depth_map * max_shift).astype(np.int32)

        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        left_x = np.clip(xx - shift_map, 0, W - 1)
        right_x = np.clip(xx + shift_map, 0, W - 1)

        left = frame[yy, left_x]
        right = frame[yy, right_x]

        # small blur to fill seams (very fast)
        left = cv2.blur(left, (3, 3))
        right = cv2.blur(right, (3, 3))

        return left, right

    def _create_anaglyph(self, left, right):
        # build red-cyan anaglyph
        anaglyph = np.zeros_like(left)
        anaglyph[:, :, 2] = left[:, :, 2]   # red from left
        anaglyph[:, :, 1] = right[:, :, 1]  # green from right
        anaglyph[:, :, 0] = right[:, :, 0]  # blue from right
        return anaglyph


# -------------------------
# UI
# -------------------------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        if Path(ICON_PATH).exists():
            self.setWindowIcon(QtGui.QIcon(ICON_PATH))
        self.setMinimumSize(560, 360)

        v = QtWidgets.QVBoxLayout(self)

        # logo image
        if Path(LOGO_PATH).exists():
            lab = QtWidgets.QLabel()
            px = QtGui.QPixmap(LOGO_PATH).scaled(220, 220, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            lab.setPixmap(px)
            lab.setAlignment(QtCore.Qt.AlignCenter)
            v.addWidget(lab)

        # controls
        self.choose_btn = QtWidgets.QPushButton("Choose Movie")
        self.choose_btn.clicked.connect(self.choose_movie)
        v.addWidget(self.choose_btn)

        self.convert_btn = QtWidgets.QPushButton("Convert to 3D (Optical-flow)")
        self.convert_btn.setEnabled(False)
        self.convert_btn.clicked.connect(self.start_conversion)
        v.addWidget(self.convert_btn)

        h = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        h.addWidget(self.progress)
        self.percent_label = QtWidgets.QLabel("0%")
        h.addWidget(self.percent_label)
        v.addLayout(h)

        self.status = QtWidgets.QLabel("Idle")
        v.addWidget(self.status)

        self.movie_path = None
        self.worker = None

    def choose_movie(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select movie", str(PROJECT_DIR), "Video Files (*.mp4 *.mkv *.mov *.avi)")
        if f:
            self.movie_path = f
            self.convert_btn.setEnabled(True)
            self.status.setText(f"Selected: {Path(f).name}")

    def start_conversion(self):
        if not self.movie_path:
            return
        # set output path
        base = Path(self.movie_path)
        out = str(base.with_name(base.stem + "_3d" + base.suffix))

        # disable UI
        self.choose_btn.setEnabled(False)
        self.convert_btn.setEnabled(False)

        # create worker
        self.worker = ConverterWorker(self.movie_path, out)
        self.worker.progress.connect(self.on_progress)
        self.worker.status.connect(lambda s: self.status.setText(s))
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
        self.status.setText("Starting conversion...")

    def on_progress(self, pct):
        self.progress.setValue(pct)
        self.percent_label.setText(f"{pct}%")

    def on_finished(self, output_path):
        self.status.setText("Finished: " + output_path)
        QtWidgets.QMessageBox.information(self, "Done", f"Saved: {output_path}")
        self.choose_btn.setEnabled(True)
        self.convert_btn.setEnabled(True)
        self.progress.setValue(100)

    def on_error(self, text):
        QtWidgets.QMessageBox.critical(self, "Error", text)
        self.status.setText("Error: " + text)
        self.choose_btn.setEnabled(True)
        self.convert_btn.setEnabled(True)


def main():
    app = QtWidgets.QApplication(sys.argv)
    # optional splash
    if Path(SPLASH_PATH).exists():
        splash_pix = QtGui.QPixmap(SPLASH_PATH)
        splash = QtWidgets.QSplashScreen(splash_pix)
        splash.show()
        QtCore.QTimer.singleShot(1400, splash.close)  # 1.4s
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
