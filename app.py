import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os, csv, io
from datetime import datetime

# ---------- PIXEL / FILTER FUNCTIONS ----------

def pixelate_gray(gray: np.ndarray, pixel_size: int) -> np.ndarray:
    """
    Turn a grayscale image into big visible square pixels by
    downscaling then upscaling with NEAREST.
    """
    h, w = gray.shape[:2]
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)

    # Downscale then upscale with NEAREST for crisp blocks
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pix

def metalwork_pixel_filter(img_bgr, pixel_size, low_thresh, high_thresh, cleanup_strength, thickness):
    """
    Better line perception:
    - bilateral denoise (keeps edges)
    - CLAHE contrast boost
    - Canny edges
    - morphology close (connect lines) + open (remove specks)
    - optional thicken
    - pixelate into big squares
    Returns a 0/255 grayscale image.
    """
    # 1) grayscale + denoise
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)

    # 2) contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3) thresholds (force sensible ordering)
    low = int(min(low_thresh, high_thresh - 1))
    high = int(max(high_thresh, low + 1))

    edges = cv2.Canny(gray, low, high)

    # 4) Connect broken linework (CLOSE) and remove specks (OPEN)
    # cleanup_strength controls how aggressively we connect lines / remove noise
    close_iters = int(cleanup_strength)
    open_iters = max(1, int(round(cleanup_strength / 2)))

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close, iterations=close_iters)

    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k_open, iterations=open_iters)

    # 5) Thickness (dilate)
    if thickness > 0:
        k = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, k, iterations=int(thickness))

    # 6) Pixelate to big squares
    pixelated = pixelate_gray(edges, pixel_size)
    return pixelated

def apply_palette(pixel_img, fg, bg):
    """
    pixel_img: 0/255 grayscale
    fg/bg are RGB tuples
    returns RGB image
    """
    out = np.zeros((pixel_img.shape[0], pixel_img.shape[1], 3), dtype=np.uint8)
    out[pixel_img > 0] = fg
    out[pixel_img == 0] = bg
    return out

# ---------- DATASET SETUP ----------

DATASET_DIR = "dataset/images"
META_PATH = "dataset/metadata.csv"
os.makedirs(DATASET_DIR, exist_ok=True)

if not os.path.exists(META_PATH):
    with open(META_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "filename", "timestamp", "pixel_size",
            "low_thresh", "high_thresh", "cleanup_strength", "thickness",
            "palette", "notes"
        ])

PALETTES = {
    "Neon Green on Black": ((0, 255, 0), (0, 0, 0)),
    "White on Black": ((255, 255, 255), (0, 0, 0)),
    "Black on White": ((0, 0, 0), (255, 255, 255)),
    "Rust on Charcoal": ((120, 60, 20), (20, 20, 20))
}

# ---------- UI ----------

st.set_page_config(layout="wide")
st.title("Global Metalwork Pixel Study")

uploaded = st.file_uploader("Upload metalwork image", type=["jpg", "png", "jpeg"])

pixel_size = st.slider("Pixel size (bigger = chunkier squares)", 4, 96, 24, step=2)

# Better defaults/ranges for avoiding background texture noise
low = st.slider("Edge low threshold", 20, 200, 80)
high = st.slider("Edge high threshold", 40, 300, 170)

# New tuning sliders
cleanup_strength = st.slider("Line cleanup / connection strength", 1, 5, 2)
thickness = st.slider("Line thickness", 0, 4, 1)

palette_name = st.selectbox("Color palette", list(PALETTES.keys()))
notes = st.text_input("Notes (region, material, source, etc.)")

if uploaded:
    # Read upload
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Process
    pixels = metalwork_pixel_filter(img_bgr, pixel_size, low, high, cleanup_strength, thickness)
    fg, bg = PALETTES[palette_name]
    final_rgb = apply_palette(pixels, fg, bg)

    final_pil = Image.fromarray(final_rgb)

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(final_pil, caption="Pixel Metalwork", use_container_width=True)

    # Download (real PNG bytes)
    buf = io.BytesIO()
    final_pil.save(buf, format="PNG")
    st.download_button(
        label="Download image",
        data=buf.getvalue(),
        file_name="metalwork_pixel.png",
        mime="image/png"
    )

    # Save to dataset
    if st.button("Save to dataset"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metal_{timestamp}.png"
        save_path = os.path.join(DATASET_DIR, filename)

        final_pil.save(save_path)

        with open(META_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                filename, timestamp, pixel_size,
                low, high, cleanup_strength, thickness,
                palette_name, notes
            ])

        st.success("Saved to dataset ✔")
