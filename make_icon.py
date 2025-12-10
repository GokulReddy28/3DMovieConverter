# make_icon.py
from PIL import Image
from pathlib import Path

# Paths
proj = Path.cwd()
assets = proj / "assets"
src = assets / "logo.png"
dst = assets / "icon.ico"

if not src.exists():
    raise SystemExit(f"Source logo not found: {src}\nPut your logo.png into the assets/ folder first.")

# Sizes Windows prefers for ICO (include 256 for modern use)
sizes = [(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)]

# Load source
im = Image.open(src).convert("RGBA")

# Ensure square canvas for best result
w, h = im.size
if w != h:
    s = max(w, h)
    square = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    square.paste(im, ((s - w) // 2, (s - h) // 2), im)
    im = square

# Save as multi-resolution ICO
im.save(dst, format="ICO", sizes=sizes)
print("Saved icon at:", dst)
