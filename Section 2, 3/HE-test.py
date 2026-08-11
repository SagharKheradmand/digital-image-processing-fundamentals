import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from HE import (
    compute_histogram_and_cdf,
    global_histogram_equalization,
    local_histogram_equalization,
    rgb_to_hsv,
    hsv_to_rgb,
    equalize_color_image_hsv,
    equalize_color_image_rgb_channels,
    compare_grayscale_results,
    compare_color_results,
    load_images_from_directory
)

INPUT_DIR = r"C:\Users\khand\Desktop\Ms.Artificial Intelegence\DIP\Projects\HW-1\DIP - HW1\Images\Section 3"
OUTPUT_DIR = r"C:\Users\khand\Desktop\Ms.Artificial Intelegence\DIP\Projects\HW-1\DIP - HW1\Results\Section 3"

#create output subdirectories
os.makedirs(os.path.join(OUTPUT_DIR, "grayscale"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "color"), exist_ok=True)

#Grayscale Histogram & CDF (For a standard image like moon.tif)
def test_histogram_and_cdf(img, filename):
    print(f"   Grayscale Histogram & CDF ({filename})   ")    
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    
    #Compute histogram and CDF
    hist, cdf = compute_histogram_and_cdf(img)
    
    print(f"\nHistogram computed: {len(hist)} bins")
    print(f"CDF computed: min={cdf.min():.2f}, max={cdf.max():.2f}")
    
    #Plot histogram
    save_path = os.path.join(OUTPUT_DIR, "grayscale", f"histogram_cdf_{filename}.png")
    
    #Create combined plot: histogram + CDF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.bar(range(256), hist, color='gray', width=1.0)
    ax1.set_title("Histogram")
    ax1.set_xlabel("Pixel Intensity")
    ax1.set_ylabel("Frequency")
    
    ax2.plot(cdf, color='blue', linewidth=2)
    ax2.set_title("Cumulative Distribution Function (CDF)")
    ax2.set_xlabel("Pixel Intensity")
    ax2.set_ylabel("CDF")
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved histogram and CDF plot to: {save_path}")


# Global Histogram Equalization (GHE) (For low-contrast image)
def test_global_histogram_equalization(img, save_path):
    print("\n   Global Histogram Equalization (GHE)   ")
    print(f"Image shape: {img.shape}")
    print(f"Intensity range: [{img.min()}, {img.max()}]")
    
    #Apply Global Histogram Equalization
    print("\nApplying Global Histogram Equalization...")
    equalized_img = global_histogram_equalization(img)
    
    print(f"Equalized intensity range: [{equalized_img.min()}, {equalized_img.max()}]")
    
    #Save comparison plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    compare_grayscale_results(img, equalized_img, method_name="Global HE", save_path=save_path)
    
    print(f"✓ Saved GHE comparison to: {save_path}")


#Local Histogram Equalization (LHE) (For hidden local details)
def test_local_histogram_equalization(img, save_path):
    print("\n   Local Histogram Equalization (LHE)   ")
    print(f"Image shape: {img.shape}")
    
    # Apply Local Histogram Equalization
    print("\nApplying Local Histogram Equalization (15x15 window)...")
    print("Warning: This may take several minutes for large images...")
    
    start_time = time.time()
    equalized_lhe = local_histogram_equalization(img, window_size=15)
    lhe_time = time.time() - start_time
    
    print(f"LHE completed in {lhe_time:.2f} seconds")
    
    #Apply GHE for comparison
    print("\nApplying Global HE for comparison...")
    start_time = time.time()
    equalized_ghe = global_histogram_equalization(img)
    ghe_time = time.time() - start_time
    
    print(f"GHE completed in {ghe_time:.4f} seconds")
    
    #Save LHE comparison
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    compare_grayscale_results(img, equalized_lhe, method_name="Local HE", save_path=save_path)
    print(f"✓ Saved LHE comparison to: {save_path}")
    
    #Create side-by-side comparison: GHE vs LHE
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original\n(Hidden Details)")
    axes[0].axis('off')
    
    axes[1].imshow(equalized_ghe, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f"Global HE\n({ghe_time:.4f}s)")
    axes[1].axis('off')
    
    axes[2].imshow(equalized_lhe, cmap='gray', vmin=0, vmax=255)
    ratio = lhe_time / ghe_time if ghe_time > 0 else 0
    axes[2].set_title(f"Local HE\n({lhe_time:.2f}s, ~{ratio:.0f}x slower)")
    axes[2].axis('off')
    
    plt.suptitle("Local HE vs Global HE: Detail Enhancement & Computational Cost", fontsize=13)
    plt.tight_layout()
    
    save_path_comparison = save_path.replace('.png', '_sidebyside.png')
    plt.savefig(save_path_comparison, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved side-by-side comparison to: {save_path_comparison}")
    print(f"   - Noise enhancement: Compare the two results visually — LHE amplifies local noise")


#Color Image Histogram Equalization (Optional)
def test_color_histogram_equalization(img_bgr, save_path):
    print("\n   Color Image Histogram Equalization   ")
    print(f"Image shape: {img_bgr.shape}")
    
    print("\nMethod 1: Applying HE to V channel (HSV method)...")
    hsv_eq_bgr = equalize_color_image_hsv(img_bgr)
    
    print("Method 2: Applying HE to R, G, B channels independently...")
    rgb_eq_bgr = equalize_color_image_rgb_channels(img_bgr)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    compare_color_results(img_bgr, hsv_eq_bgr, rgb_eq_bgr, save_path=save_path)
    
    print(f"✓ Saved color HE comparison to: {save_path}")

#Main
if __name__ == "__main__":
    
    # 1. Test Histogram & CDF on moon.tif
    moon_path = os.path.join(INPUT_DIR, "moon.tif")
    img_moon = cv2.imread(moon_path, cv2.IMREAD_GRAYSCALE)
    if img_moon is not None:
        test_histogram_and_cdf(img_moon, "moon.tif")
    else:
        print(f"Could not load {moon_path}")

    # 2. Test GHE on einstein.jpg (Low contrast)
    einstein_path = os.path.join(INPUT_DIR, "einstein.jpg")
    img_einstein = cv2.imread(einstein_path, cv2.IMREAD_GRAYSCALE)
    if img_einstein is not None:
        save_path = os.path.join(OUTPUT_DIR, "grayscale", "ghe_einstein.png")
        test_global_histogram_equalization(img_einstein, save_path)
    else:
        print(f"Could not load {einstein_path}")

    # 3. Test LHE on med1.jpg (Hidden details)
    med_path = os.path.join(INPUT_DIR, "med1.jpg")
    img_med = cv2.imread(med_path, cv2.IMREAD_GRAYSCALE)
    if img_med is not None:
        save_path = os.path.join(OUTPUT_DIR, "grayscale", "lhe_med1.png")
        test_local_histogram_equalization(img_med, save_path)
    else:
        print(f"Could not load {med_path}")

    # 4. Optional: Run Color HE 
    peppers_path = os.path.join(INPUT_DIR, "mandrill.png")
    img_peppers = cv2.imread(peppers_path) # Read as BGR color
    if img_peppers is not None:
        save_path = os.path.join(OUTPUT_DIR, "color", "color_mandrill.png")
        test_color_histogram_equalization(img_peppers, save_path)
