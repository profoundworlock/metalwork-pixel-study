import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os, csv
from datetime import datetime

# ---------- FILTER FUNCTIONS ----------

def metalwork_pixel_filter(img_bgr, pixel_size, low_thresh, high_thresh):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, low_thresh, high_thresh)

    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    _, binary = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)

    h, w = binary.shape
    small = cv2.resize(binary, (w//pixel_size, h//pixel_size))
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return pixelated

def apply_palette(pixel_img, fg, bg):
    color = np.zeros((*pixel_img.shape, 3), dtype=np.uint8)
    color[pixel_img > 0] = fg
    color[pixel_img == 0] = bg
    return color

# ---------- DATASET SETUP ----------

DATASET_DIR = "dataset/images"
META_PATH = "dataset/metadata.csv"
os.makedirs(DATASET_DIR, exist_ok=True)

if not os.path.exists(META_PATH):
    with open(META_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "filename","timestamp","pixel_size",
            "low_thresh","high_thresh","palette","notes"
        ])

PALETTES = {
    "Neon Green on Black": ((0,255,0), (0,0,0)),
    "White on Black": ((255,255,255), (0,0,0)),
    "Black on White": ((0,0,0), (255,255,255)),
    "Rust on Charcoal": ((120,60,20), (20,20,20))
}

# ---------- UI ----------

st.set_page_config(layout="wide")
st.title("Global Metalwork Pixel Study")

uploaded = st.file_uploader("Upload metalwork image", type=["jpg","png","jpeg"])

pixel_size = st.slider("Pixel size", 4, 64, 16, step=2)
low = st.slider("Edge low threshold", 10, 150, 60)
high = st.slider("Edge high threshold", 50, 300, 140)

palette_name = st.selectbox("Color palette", list(PALETTES.keys()))
notes = st.text_input("Notes (region, material, source, etc.)")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    pixels = metalwork_pixel_filter(img_bgr, pixel_size, low, high)
    fg, bg = PALETTES[palette_name]
    final = apply_palette(pixels, fg, bg)

    st.image([image, final], caption=["Original", "Pixel Metalwork"], width=400)

    result_pil = Image.fromarray(final)

    st.download_button(
        label="Download image",
        data=result_pil.tobytes(),
        file_name="metalwork_pixel.png",
        mime="image/png"
    )

    if st.button("Save to dataset"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metal_{timestamp}.png"
        save_path = os.path.join(DATASET_DIR, filename)

        result_pil.save(save_path)

        with open(META_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                filename, timestamp, pixel_size,
                low, high, palette_name, notes
            ])

        st.success("Saved to dataset ✔")
