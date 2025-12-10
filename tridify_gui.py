import os
import sys
import subprocess
import threading
import time
from tkinter import *
from tkinter import filedialog, messagebox, ttk


# ============================================================
# AUTO-DETECT ENGINE EXE (Fix for "Engine EXE not found!")
# ============================================================
def get_engine_path():
    """
    Detect engine exe in the SAME FOLDER as tridify_gui.exe.
    """
    # When running as EXE
    exe_dir = os.path.dirname(sys.argv[0])

    possible_engines = [
        "convert_movie_to_3d_hybrid_audio.exe",
        "convert_movie_to_3d_hybrid.exe",
    ]

    # Look in the EXE directory first
    for exe_name in possible_engines:
        full_path = os.path.join(exe_dir, exe_name)
        if os.path.exists(full_path):
            return full_path

    # If not found, fallback to MEIPASS (PyInstaller temp folder)
    base = getattr(sys, "_MEIPASS", "")
    for exe_name in possible_engines:
        full_path = os.path.join(base, exe_name)
        if os.path.exists(full_path):
            return full_path

    return ""



# ============================================================
# GUI APPLICATION
# ============================================================

class TriDifyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TriDify – 3D Movie Converter")
        self.root.geometry("1000x650")
        self.root.configure(bg="#1A1A1A")

        # ===== VARIABLES =====
        self.input_file = StringVar()
        self.output_file = StringVar()
        self.engine_file = StringVar()

        # Auto-fill engine path
        detected_engine = get_engine_path()
        self.engine_file.set(detected_engine)

        # ====================================================
        # TITLE BAR
        # ====================================================
        title = Label(
            root,
            text="TriDify",
            fg="#FFFFFF",
            bg="#1A1A1A",
            font=("Arial", 40, "bold")
        )
        title.pack(pady=10)

        subtitle = Label(
            root,
            text="Convert any 2D movie into 3D (Hybrid + Audio)",
            fg="#CCCCCC",
            bg="#1A1A1A",
            font=("Arial", 18)
        )
        subtitle.pack()

        # ====================================================
        # INPUT FRAME
        # ====================================================
        frame = Frame(root, bg="#1A1A1A")
        frame.pack(pady=30)

        Label(frame, text="Input Video:", fg="#FFFFFF", bg="#1A1A1A", font=("Arial", 16)).grid(row=0, column=0, sticky="w")
        Entry(frame, textvariable=self.input_file, width=60).grid(row=1, column=0, padx=10)
        Button(frame, text="Browse Input", command=self.browse_input).grid(row=1, column=1, padx=10)

        Label(frame, text="Output File:", fg="#FFFFFF", bg="#1A1A1A", font=("Arial", 16)).grid(row=2, column=0, sticky="w", pady=20)
        Entry(frame, textvariable=self.output_file, width=60).grid(row=3, column=0, padx=10)
        Button(frame, text="Browse Output", command=self.browse_output).grid(row=3, column=1, padx=10)

        Label(frame, text="TriDify Engine (EXE):", fg="#FFFFFF", bg="#1A1A1A", font=("Arial", 16)).grid(row=4, column=0, sticky="w", pady=20)
        Entry(frame, textvariable=self.engine_file, width=60).grid(row=5, column=0, padx=10)
        Button(frame, text="Locate Engine", command=self.browse_engine).grid(row=5, column=1, padx=10)

        # ====================================================
        # PROGRESS SECTION
        # ====================================================
        progress_frame = Frame(root, bg="#1A1A1A")
        progress_frame.pack()

        Label(progress_frame, text="Progress", fg="#FFFFFF", bg="#1A1A1A", font=("Arial", 22, "bold")).pack()

        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode="determinate")
        self.progress_bar.pack(pady=10)

        self.percent_label = Label(progress_frame, text="0%", fg="#FFFFFF", bg="#1A1A1A", font=("Arial", 14))
        self.percent_label.pack()

        self.eta_label = Label(progress_frame, text="ETA: --", fg="#AAAAAA", bg="#1A1A1A", font=("Arial", 12))
        self.eta_label.pack()

        # ====================================================
        # CONTROL BUTTONS
        # ====================================================
        btn_frame = Frame(root, bg="#1A1A1A")
        btn_frame.pack(pady=25)

        Button(btn_frame, text="Start Conversion", bg="#28A745", fg="white",
               font=("Arial", 14), width=15, command=self.start_conversion).grid(row=0, column=0, padx=15)

        Button(btn_frame, text="Cancel", bg="#DC3545", fg="white",
               font=("Arial", 14), width=12, command=self.cancel).grid(row=0, column=1, padx=15)

        Button(btn_frame, text="Preview Output", bg="#007BFF", fg="white",
               font=("Arial", 14), width=15, command=self.preview_output).grid(row=0, column=2, padx=15)

        self.process = None

    # ============================================================
    # BROWSE BUTTON ACTIONS
    # ============================================================

    def browse_input(self):
        f = filedialog.askopenfilename(title="Select a video file")
        if f:
            self.input_file.set(f)

    def browse_output(self):
        f = filedialog.asksaveasfilename(title="Save output video", defaultextension=".mp4")
        if f:
            self.output_file.set(f)

    def browse_engine(self):
        f = filedialog.askopenfilename(title="Select Engine EXE", filetypes=[("EXE files", "*.exe")])
        if f:
            self.engine_file.set(f)

    # ============================================================
    # MAIN CONVERSION LOGIC
    # ============================================================

    def start_conversion(self):
        engine = self.engine_file.get().strip()
        if not os.path.exists(engine):
            messagebox.showerror("Error", "Engine not found! Please select a valid EXE.")
            return

        inp = self.input_file.get().strip()
        out = self.output_file.get().strip()

        if not inp or not out:
            messagebox.showerror("Error", "Please select input and output files.")
            return

        cmd = [engine, inp, out]

        def run_process():
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

            for line in self.process.stdout:
                self.update_progress(line)

            self.process.wait()
            messagebox.showinfo("Done", "3D Conversion Completed!")

        threading.Thread(target=run_process, daemon=True).start()

    def update_progress(self, text):
        if "PROG:" in text:
            try:
                percent = int(text.split("PROG:")[1].split("%")[0].strip())
                self.progress_bar["value"] = percent
                self.percent_label.config(text=f"{percent}%")
            except:
                pass

        if "ETA:" in text:
            eta = text.split("ETA:")[1].strip()
            self.eta_label.config(text=f"ETA: {eta}")

    # Cancel conversion
    def cancel(self):
        if self.process:
            self.process.terminate()
            messagebox.showinfo("Stopped", "Conversion cancelled.")

    # Preview output video
    def preview_output(self):
        out = self.output_file.get().strip()
        if os.path.exists(out):
            os.startfile(out)


# ============================================================
# START APP
# ============================================================

if __name__ == "__main__":
    root = Tk()
    app = TriDifyApp(root)
    root.mainloop()
