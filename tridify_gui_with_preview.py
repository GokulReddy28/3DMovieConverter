# tridify_gui_with_preview.py
import os
import subprocess
import threading
import time
import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox, Toplevel, Label, Button
from PIL import Image, ImageTk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class TriDifyApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("TriDify – 3D Movie Converter")
        self.geometry("1000x650")
        self.resizable(False, False)

        self.process = None
        self.preview_running = False
        self.preview_window = None
        self.preview_label = None
        self._build_ui()

    # ---------------------------------------------------------
    # UI BUILD
    # ---------------------------------------------------------
    def _build_ui(self):
        # TITLE AREA
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(pady=10)

        ctk.CTkLabel(header, text="TriDify",
                     font=("Segoe UI", 38, "bold")).pack()

        ctk.CTkLabel(header,
                     text="Convert any 2D movie into 3D (Hybrid + Audio)",
                     font=("Segoe UI", 18)).pack()

        # MAIN 2-COLUMN LAYOUT
        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=20, pady=15)

        left = ctk.CTkFrame(main, corner_radius=12)
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        right = ctk.CTkFrame(main, corner_radius=12)
        right.pack(side="right", fill="y", padx=10, pady=10)

        # LEFT INPUT SECTION
        ctk.CTkLabel(left, text="Input Video:", font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(10, 0))

        self.input_entry = ctk.CTkEntry(left, placeholder_text="Choose input video…", width=500)
        self.input_entry.pack(pady=5, padx=10)

        ctk.CTkButton(left, text="Browse Input", width=140, command=self._browse_input).pack(pady=5)

        # Output
        ctk.CTkLabel(left, text="Output File:", font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(20, 0))

        self.output_entry = ctk.CTkEntry(left, placeholder_text="Choose output 3D video path…", width=500)
        self.output_entry.pack(pady=5, padx=10)

        ctk.CTkButton(left, text="Browse Output", width=140, command=self._browse_output).pack(pady=5)

        # EXE PATH
        ctk.CTkLabel(left, text="TriDify Engine (EXE):", font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(20, 0))

        self.exe_entry = ctk.CTkEntry(left, width=500)
        # default path suggestion (change if needed)
        self.exe_entry.insert(0, os.path.join(os.getcwd(), "dist", "convert_movie_to_3d_hybrid.exe"))
        self.exe_entry.pack(padx=10, pady=5)

        ctk.CTkButton(left, text="Locate Engine", width=140, command=self._browse_exe).pack(pady=5)

        # OPTIONS
        self.open_after = ctk.CTkCheckBox(left, text="Open output folder when finished")
        # select by default:
        try:
            # CTkCheckBox in current versions supports .select()
            self.open_after.select()
        except Exception:
            pass
        self.open_after.pack(anchor="w", pady=10, padx=10)

        # -----------------------------------------------
        # RIGHT PANEL (CONTROLS + PROGRESS)
        # -----------------------------------------------
        ctk.CTkLabel(right, text="Progress", font=("Segoe UI", 20, "bold")).pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(right, width=260)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10)

        self.progress_label = ctk.CTkLabel(right, text="0%", font=("Segoe UI", 15))
        self.progress_label.pack()

        self.start_btn = ctk.CTkButton(right, text="Start Conversion", fg_color="green",
                                       width=200, height=40, command=self._start_conversion)
        self.start_btn.pack(pady=10)

        self.cancel_btn = ctk.CTkButton(right, text="Cancel", fg_color="red",
                                        width=200, height=40, command=self._cancel_conversion)
        self.cancel_btn.pack(pady=5)

        # Preview button (disabled until output exists)
        self.preview_btn = ctk.CTkButton(right, text="Preview Output", width=200, height=40,
                                         command=self._open_preview, state="disabled")
        self.preview_btn.pack(pady=(20,5))

        # -----------------------------------------------
        # LOG SECTION
        # -----------------------------------------------
        ctk.CTkLabel(self, text="Console Log:", font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=20)

        log_frame = ctk.CTkFrame(self)
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.console = ctk.CTkTextbox(log_frame, height=180)
        self.console.pack(fill="both", expand=True)

    # ---------------------------------------------------------
    # DIALOG FUNCTIONS
    # ---------------------------------------------------------
    def _browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.mkv *.mov *.avi")])
        if path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 video","*.mp4")])
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)

    def _browse_exe(self):
        path = filedialog.askopenfilename(filetypes=[("Executable", "*.exe")])
        if path:
            self.exe_entry.delete(0, "end")
            self.exe_entry.insert(0, path)

    # ---------------------------------------------------------
    # LOG
    # ---------------------------------------------------------
    def _log(self, text):
        self.console.configure(state="normal")
        self.console.insert("end", text + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    # ---------------------------------------------------------
    # START CONVERSION
    # ---------------------------------------------------------
    def _start_conversion(self):
        exe = self.exe_entry.get().strip()
        inp = self.input_entry.get().strip()
        out = self.output_entry.get().strip()

        if not os.path.exists(exe):
            messagebox.showerror("Error", "Engine EXE not found!")
            return
        if not os.path.exists(inp):
            messagebox.showerror("Error", "Input video missing!")
            return
        if out == "":
            messagebox.showerror("Error", "Please choose an output file path.")
            return

        # reset UI
        self.progress_bar.set(0)
        self.progress_label.configure(text="0%")
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")
        self.preview_btn.configure(state="disabled")

        self._log("Starting Conversion...")
        self._log(f"Engine: {exe}")
        self._log(f"Input: {inp}")
        self._log(f"Output: {out}")

        # start process
        try:
            self.process = subprocess.Popen(
                [exe, inp, out],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start engine: {e}")
            return

        # start thread to read process output
        threading.Thread(target=self._read_output, daemon=True).start()

    # ---------------------------------------------------------
    # READ PROCESS OUTPUT
    # ---------------------------------------------------------
    def _read_output(self):
        # read stdout line-by-line and update log & progress
        for line in self.process.stdout:
            line = line.rstrip("\n")
            self._log(line)

            # try to parse percentage from common patterns like "xx%" or "Progress: xx%"
            if "%" in line:
                try:
                    # find first integer in line
                    digits = "".join(ch for ch in line if ch.isdigit())
                    if digits:
                        num = int(digits)
                        if 0 <= num <= 100:
                            self.progress_bar.set(num / 100.0)
                            self.progress_label.configure(text=f"{num}%")
                except Exception:
                    pass

        self.process.wait()
        ret = self.process.returncode

        if ret == 0:
            self._log("\n✔ Conversion Completed Successfully!")
            # enable preview button if output exists
            outpath = self.output_entry.get().strip()
            if os.path.exists(outpath):
                self.preview_btn.configure(state="normal")
            # open folder if requested
            try:
                if self.open_after.get():
                    folder = os.path.dirname(outpath) or "."
                    try:
                        os.startfile(folder)
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            self._log("\n❌ Conversion Failed (exit code {})".format(ret))

        self.process = None

    # ---------------------------------------------------------
    # CANCEL
    # ---------------------------------------------------------
    def _cancel_conversion(self):
        if self.process:
            try:
                self.process.kill()
            except Exception:
                pass
            self._log("❌ CANCELLED by user")
            self.progress_bar.set(0)
            self.progress_label.configure(text="0%")
            self.process = None

    # ---------------------------------------------------------
    # Preview (Requirement #3 only)
    # ---------------------------------------------------------
    def _open_preview(self):
        outpath = self.output_entry.get().strip()
        if not outpath or not os.path.exists(outpath):
            messagebox.showerror("Error", "Output file not found to preview.")
            return

        # If preview already open, bring to front
        if self.preview_window and self.preview_running:
            try:
                self.preview_window.lift()
            except Exception:
                pass
            return

        # create a Toplevel window for preview playback
        self.preview_window = Toplevel(self)
        self.preview_window.title("Preview - " + os.path.basename(outpath))
        self.preview_window.geometry("600x400")
        self.preview_window.protocol("WM_DELETE_WINDOW", self._stop_preview)

        # placeholder label where frames will be shown
        self.preview_label = Label(self.preview_window)
        self.preview_label.pack(expand=True)

        # small control row
        btn_frame = ctk.CTkFrame(self.preview_window)
        btn_frame.pack(fill="x", pady=6)
        preview_openbtn = Button(btn_frame, text="Open in default player", command=lambda: self._open_in_system(outpath))
        preview_openbtn.pack(side="left", padx=8, pady=6)
        close_btn = Button(btn_frame, text="Close Preview", command=self._stop_preview)
        close_btn.pack(side="right", padx=8, pady=6)

        # start playback thread
        self.preview_running = True
        threading.Thread(target=self._play_video_thread, args=(outpath,), daemon=True).start()

    def _open_in_system(self, path):
        try:
            os.startfile(path)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open file: {e}")

    def _stop_preview(self):
        self.preview_running = False
        # let the playback thread close the window
        # but also try to close if still present
        if self.preview_window:
            try:
                self.preview_window.destroy()
            except Exception:
                pass
        self.preview_window = None
        self.preview_label = None

    def _play_video_thread(self, path):
        """
        Playback loop that reads frames using OpenCV and updates the Tk label.
        Runs in a background thread.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._log("Preview: cannot open video for preview.")
            self.preview_running = False
            try:
                if self.preview_window:
                    self.preview_window.destroy()
            finally:
                self.preview_window = None
            return

        # target display size
        target_w, target_h = 560, 320

        while self.preview_running:
            ret, frame = cap.read()
            if not ret:
                # loop playback: seek to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.1)
                continue

            # convert BGR->RGB and resize for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            scale = min(target_w / w, target_h / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # update label (on main thread)
            try:
                if self.preview_label:
                    self.preview_label.imgtk = imgtk  # prevent GC
                    self.preview_label.configure(image=imgtk)
                else:
                    break
            except Exception:
                break

            # small sleep to match source fps approximately
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            time.sleep(max(0.01, 1.0 / fps))

        cap.release()
        # close preview window if still open
        try:
            if self.preview_window:
                self.preview_window.destroy()
        except Exception:
            pass
        self.preview_window = None
        self.preview_label = None
        self.preview_running = False

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    app = TriDifyApp()
    app.mainloop()
