import os
import cv2
import numpy as np
from PO import (
    image_negative,
    log_transform,
    gamma_correction,
    piecewise_linear,
    bit_plane_slicing,
    plot_transformation
)

# ---------------------------------------------------------
# 1) Your Mapping Dictionary
# ---------------------------------------------------------
specific_images = {
    "log_transform": "med1.jpg",
    "gamma_correction": ["low contrast.jpg", "trees.tif"],  # multiple images
    "piecewise_linear": "low contrast.jpg",
    "bit_plane_slicing": "einstein.jpg"
}

# ---------------------------------------------------------
# Helper: Load all images from folder
# ---------------------------------------------------------
def load_images(folder_path):
    images = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                images[filename] = img
                print(f"Loaded: {filename} - Shape: {img.shape}")
    return images

# ---------------------------------------------------------
# Helper: Save any result image
# ---------------------------------------------------------
def save_result(img, output_folder, filename):
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, filename)
    cv2.imwrite(path, img)
    print(f"Saved: {path}")

# ---------------------------------------------------------
# Helper: Load a specific image by name
# ---------------------------------------------------------
def load_specific_image(folder, filename):
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    print(f"⚠ Specific image not found or unreadable: {path}")
    return None

# ---------------------------------------------------------
# Optional grayscale convert
# ---------------------------------------------------------
def to_grayscale(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ---------------------------------------------------------
# MAIN: Test all point operations
# ---------------------------------------------------------
def test_all_operations(input_folder, output_folder):

    images = load_images(input_folder)
    if not images:
        print("No images found.")
        return

    r = np.arange(256)

    for img_name, img in images.items():
        print(f"\nProcessing: {img_name}")
        base = os.path.splitext(img_name)[0]

        # -------------------------------------------------
        # 1) Image Negative (always uses current image)
        # -------------------------------------------------
        neg = image_negative(img)
        save_result(neg, f"{output_folder}/negative", f"{base}_negative.png")

        plot_transformation(
            r, 255 - r,
            img, neg,
            "Image Negative",
            save_path=f"{output_folder}/negative/{base}_negative_plot.png"
        )

    # -----------------------------------------------------
    # 2) LOG TRANSFORM (specific image ONLY)
    # -----------------------------------------------------
    print("\n=== LOG TRANSFORM ===")
    log_filename = specific_images["log_transform"]
    log_src = load_specific_image(input_folder, log_filename)

    if log_src is not None:
        log_img = log_transform(log_src)
        save_result(log_img, f"{output_folder}/log", "log_med1.png")

        c = 255 / np.log(1 + 255)
        plot_transformation(
            r, c * np.log(1 + r),
            log_src, log_img,
            "Log Transform",
            save_path=f"{output_folder}/log/log_med1_plot.png"
        )

    # -----------------------------------------------------
    # 3) GAMMA CORRECTION (multiple specific images)
    # -----------------------------------------------------
    print("\n=== GAMMA CORRECTION ===")

    gamma_list = specific_images["gamma_correction"]

    # Convert to list if user provided a string
    if isinstance(gamma_list, str):
        gamma_list = [gamma_list]

    for fname in gamma_list:
        gamma_src = load_specific_image(input_folder, fname)
        if gamma_src is None:
            continue

        base = os.path.splitext(fname)[0]

        for g in [0.5, 1.5, 2.5]:
            g_img = gamma_correction(gamma_src, g)
            save_result(g_img, f"{output_folder}/gamma", f"{base}_gamma_{g}.png")

            plot_transformation(
                r, np.clip(((r / 255) ** g) * 255, 0, 255),
                gamma_src, g_img,
                f"Gamma Correction γ={g}",
                save_path=f"{output_folder}/gamma/{base}_gamma_{g}_plot.png"
            )

    # -----------------------------------------------------
    # 4) PIECEWISE LINEAR (specific image)
    # -----------------------------------------------------
    print("\n=== PIECEWISE LINEAR ===")

    pw_filename = specific_images["piecewise_linear"]
    pw_src = load_specific_image(input_folder, pw_filename)

    if pw_src is not None:
        base = os.path.splitext(pw_filename)[0]

        pw = piecewise_linear(pw_src, r1=50, s1=30, r2=200, s2=220)
        save_result(pw, f"{output_folder}/piecewise", f"{base}_piecewise.png")

        s_pw = np.piecewise(
            r.astype(float),
            [r < 50, (r >= 50) & (r <= 200), r > 200],
            [
                lambda x: (30/50)*x,
                lambda x: 30 + ((220-30)/(200-50))*(x-50),
                lambda x: 220 + ((255-220)/(255-200))*(x-200)
            ]
        )

        plot_transformation(
            r, s_pw,
            pw_src, pw,
            "Piecewise Linear",
            save_path=f"{output_folder}/piecewise/{base}_piecewise_plot.png"
        )


# ---------------------------------------------------------
# BIT‑PLANE SLICING (specific image)
# ---------------------------------------------------------
def test_bit_plane_slicing(input_folder, output_folder):

    print("\n=== BIT-PLANE SLICING ===")

    bp_filename = specific_images["bit_plane_slicing"]
    img = load_specific_image(input_folder, bp_filename)

    if img is None:
        print("Skipping bit-plane slicing.")
        return

    img_gray = to_grayscale(img)
    out_folder = os.path.join(output_folder, "bit_plane_slicing")

    msb_recon, lsb_recon = bit_plane_slicing(
        img_gray,
        save_path=os.path.join(out_folder, f"{bp_filename}.png")
    )

    base = os.path.splitext(bp_filename)[0]
    save_result(msb_recon, out_folder, f"{base}_msb_recon.png")
    save_result(lsb_recon, out_folder, f"{base}_lsb_recon.png")

    print(f"Bit-plane slicing complete → {out_folder}")


# ---------------------------------------------------------
# RUN EVERYTHING
# ---------------------------------------------------------
if __name__ == "__main__":
    input_folder = r"C:\Users\khand\Desktop\Ms.Artificial Intelegence\DIP\Projects\HW-1\DIP - HW1\Images\Section 2"
    output_folder = r"C:\Users\khand\Desktop\Ms.Artificial Intelegence\DIP\Projects\HW-1\DIP - HW1\Results\Section 2"

    test_all_operations(input_folder, output_folder)
    test_bit_plane_slicing(input_folder, output_folder)
