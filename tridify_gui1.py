import os
import subprocess
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
import re

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class TriDifyApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("TriDify – 3D Movie Converter")
        self.geometry("1000x650")
        self.resizable(False, False)

        self.process = None
        self.total_frames = None

        self._build_ui()

    # ---------------------------------------------------------
    # UI BUILD
    # ---------------------------------------------------------
    def _build_ui(self):

        # TITLE AREA
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(pady=10)

        ctk.CTkLabel(
            header,
            text="TriDify",
            font=("Segoe UI", 38, "bold")
        ).pack()

        ctk.CTkLabel(
            header,
            text="Convert any 2D movie into 3D (Hybrid + Audio)",
            font=("Segoe UI", 18)
        ).pack()

        # MAIN 2-COLUMN LAYOUT
        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=20, pady=15)

        left = ctk.CTkFrame(main, corner_radius=12)
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        right = ctk.CTkFrame(main, corner_radius=12)
        right.pack(side="right", fill="y", padx=10, pady=10)

        # -----------------------------------------------
        # LEFT INPUT SECTION
        # -----------------------------------------------
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
        self.exe_entry.insert(0, "C:\\3DProject\\dist\\convert_movie_to_3d_hybrid.exe")
        self.exe_entry.pack(padx=10, pady=5)
        ctk.CTkButton(left, text="Locate Engine", width=140, command=self._browse_exe).pack(pady=5)

        # OPTIONS
        self.open_after = ctk.CTkCheckBox(left, text="Open output folder when finished")
        self.open_after.select()
        self.open_after.pack(anchor="w", pady=10, padx=10)

        # -----------------------------------------------
        # RIGHT PANEL (CONTROLS + PROGRESS)
        # -----------------------------------------------
        ctk.CTkLabel(right, text="Progress", font=("Segoe UI", 20, "bold")).pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(right, width=260)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=15)

        self.progress_label = ctk.CTkLabel(right, text="0%", font=("Segoe UI", 15))
        self.progress_label.pack()

        self.start_btn = ctk.CTkButton(right, text="Start Conversion", fg_color="green",
                                       width=200, height=40, command=self._start_conversion)
        self.start_btn.pack(pady=20)

        self.cancel_btn = ctk.CTkButton(right, text="Cancel", fg_color="red",
                                        width=200, height=40, command=self._cancel_conversion)
        self.cancel_btn.pack(pady=5)

        # -----------------------------------------------
        # LOG SECTION
        # -----------------------------------------------
        ctk.CTkLabel(self, text="Console Log:", font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=20)

        log_frame = ctk.CTkFrame(self)
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.console = ctk.CTkTextbox(log_frame, height=180)
        self.console.pack(fill="both", expand=True)

    # ---------------------------------------------------------
    # Dialog functions
    # ---------------------------------------------------------
    def _browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.mkv *.mov *.avi")])
        if path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4")
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

        self.progress_bar.set(0)
        self.progress_label.configure(text="0%")

        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

        self._log("Starting Conversion...")
        self._log(f"Engine: {exe}")
        self._log(f"Input: {inp}")
        self._log(f"Output: {out}")

        self.total_frames = None

        self.process = subprocess.Popen(
            [exe, inp, out],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        threading.Thread(target=self._read_output, daemon=True).start()

    # ---------------------------------------------------------
    # READ OUTPUT (PROGRESS PARSING)
    # ---------------------------------------------------------
    def _read_output(self):

        for line in self.process.stdout:
            line = line.strip()
            self._log(line)

            # Capture frame count
            if "frames=" in line:
                match = re.search(r"frames=(\d+)", line)
                if match:
                    self.total_frames = int(match.group(1))

            # Capture keyframe progress
            if "Keyframe at" in line and self.total_frames:
                match = re.search(r"Keyframe at (\d+)", line)
                if match:
                    frame_num = int(match.group(1))
                    pct = min(100, int((frame_num / self.total_frames) * 100))
                    self.progress_bar.set(pct / 100)
                    self.progress_label.configure(text=f"{pct}%")

            # Detect finish
            if "Done." in line:
                self.progress_bar.set(1)
                self.progress_label.configure(text="100%")

        self.process.wait()

        # Completed
        if self.process.returncode == 0:
            self._log("\n✔ Conversion Completed Successfully!")

            if self.open_after.get():
                try:
                    os.startfile(os.path.dirname(self.output_entry.get()))
                except:
                    pass
        else:
            self._log("\n❌ Conversion Failed")

        self.process = None

    # ---------------------------------------------------------
    # CANCEL
    # ---------------------------------------------------------
    def _cancel_conversion(self):
        if self.process:
            self.process.kill()
            self._log("❌ CANCELLED by user")
            self.progress_bar.set(0)
            self.progress_label.configure(text="0%")
            self.process = None


if __name__ == "__main__":
    app = TriDifyApp()
    app.mainloop()
