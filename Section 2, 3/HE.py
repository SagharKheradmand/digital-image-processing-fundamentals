import numpy as np
import matplotlib.pyplot as plt
import cv2  
import os

def load_images_from_directory(directory):

    gray_images = []
    color_images = []
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(valid_extensions):
            filepath = os.path.join(directory, filename)
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                print(f"Warning: Could not read {filename}")
                continue
            
            #check if grayscale or color
            if len(img.shape) == 2:
                gray_images.append((filename, img))
            elif len(img.shape) == 3 and img.shape[2] == 3:
                color_images.append((filename, img))
    
    return gray_images, color_images

#Histogram & CDF 
def compute_histogram_and_cdf(img):

    hist = np.zeros(256, dtype=int)  #initialize histogram array with zeros
    rows, cols = img.shape           #get image dimensions

    #manually count each pixel's intensity
    for r in range(rows):
        for c in range(cols):
            intensity = img[r, c]                    #read pixel value (0–255)
            hist[intensity] += 1                     #increment the corresponding bin

    total_pixels = rows * cols                       #total number of pixels
    pdf = hist / total_pixels                        #normalize histogram to get PDF

    #compute CDF as cumulative sum of PDF
    cdf = np.zeros(256, dtype=float)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]                 #each CDF value = previous + current PDF

    return hist, cdf

#Plots Histogram
def plot_histogram(hist, title="Histogram", save_path=None):

    plt.figure(figsize=(6, 4))
    plt.bar(range(256), hist, color='gray', width=1.0)  # bar chart of intensities
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)  # save to file
        plt.close()                                            # free memory
    else:
        plt.show()


#Global Histogram Equalization
def global_histogram_equalization(img):

    hist, cdf = compute_histogram_and_cdf(img)             #compute histogram and CDF

    #build the transformation mapping: s = round(255 * CDF(r))
    mapping = np.round(255 * cdf).astype(np.uint8)         #one mapped value per intensity level

    #apply the mapping: each pixel's value is replaced by its mapped value
    equalized_img = mapping[img]

    return equalized_img


#Local Histogram Equalization
def local_histogram_equalization(img, window_size=15):

    rows, cols = img.shape                         #get image dimensions
    pad = window_size // 2                         #padding size to handle image borders

    #pad image with reflected values to avoid border artifacts
    padded_img = np.pad(img, pad, mode='reflect')

    output_img = np.zeros_like(img)                #output image, same size as input

    window_area = window_size * window_size         #total pixels in each window

    for r in range(rows):
        for c in range(cols):
            #extract the local neighborhood window around pixel (r, c)
            window = padded_img[r:r + window_size, c:c + window_size]

            #manually compute local histogram (no np.histogram allowed)
            local_hist = np.zeros(256, dtype=int)
            for wr in range(window_size):
                for wc in range(window_size):
                    local_hist[window[wr, wc]] += 1               #count each intensity in window

            #get the center pixel's intensity value
            center_val = img[r, c]

            #compute cumulative sum of local histogram up to center pixel's value
            #this gives us the local CDF at that intensity
            cumulative_sum = 0
            for i in range(center_val + 1):
                cumulative_sum += local_hist[i]

            #normalize to get local CDF value
            local_cdf = cumulative_sum / window_area

            #apply equalization mapping: s = round(255 * local_CDF(r))
            output_img[r, c] = np.round(255 * local_cdf)

    return output_img


#COLOR IMAGE SUPPORT — RGB ↔ HSV
def rgb_to_hsv(img_rgb):

    #normalize RGB to [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]

    Cmax = np.max(img_float, axis=2)                #maximum of R, G, B per pixel
    Cmin = np.min(img_float, axis=2)                #minimum of R, G, B per pixel
    delta = Cmax - Cmin                             #chroma (range of RGB values)

    #Value channel: V = Cmax 
    V = Cmax

    #Saturation channel: S = delta / Cmax (0 if Cmax == 0)
    S = np.where(Cmax == 0, 0.0, delta / Cmax)

    #Hue channel: angle in [0, 360]
    H = np.zeros_like(V)

    #when R is max
    mask_r = (Cmax == R) & (delta != 0)
    H[mask_r] = (60 * ((G[mask_r] - B[mask_r]) / delta[mask_r])) % 360

    #when G is max
    mask_g = (Cmax == G) & (delta != 0)
    H[mask_g] = 60 * ((B[mask_g] - R[mask_g]) / delta[mask_g] + 2)

    #when B is max
    mask_b = (Cmax == B) & (delta != 0)
    H[mask_b] = 60 * ((R[mask_b] - G[mask_b]) / delta[mask_b] + 4)

    #stack into a single HSV image
    hsv = np.stack([H, S, V], axis=2)
    return hsv


def hsv_to_rgb(hsv):

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    #intermediate values for the HSV → RGB formula
    C = V * S                                               #chroma
    X = C * (1 - np.abs((H / 60) % 2 - 1))                  #second largest component
    m = V - C                                               #match value to shift RGB up

    #initialize RGB channels
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    #assign R, G, B based on which 60° sector H falls in
    mask = (H >= 0)   & (H < 60);  R[mask] = C[mask]; G[mask] = X[mask]
    mask = (H >= 60)  & (H < 120); R[mask] = X[mask]; G[mask] = C[mask]
    mask = (H >= 120) & (H < 180); G[mask] = C[mask]; B[mask] = X[mask]
    mask = (H >= 180) & (H < 240); G[mask] = X[mask]; B[mask] = C[mask]
    mask = (H >= 240) & (H < 300); R[mask] = X[mask]; B[mask] = C[mask]
    mask = (H >= 300) & (H < 360); R[mask] = C[mask]; B[mask] = X[mask]

    #add m to shift all channels, then scale to [0, 255]
    img_rgb = np.stack([R + m, G + m, B + m], axis=2)
    img_rgb = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)

    return img_rgb


def equalize_color_image_hsv(img_bgr):

    #convert BGR (OpenCV default) to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    #convert RGB to HSV using our from-scratch function
    hsv = rgb_to_hsv(img_rgb)

    #extract V channel and scale to uint8 for equalization
    V_channel = (hsv[:, :, 2] * 255).astype(np.uint8)

    #apply global histogram equalization to V channel only
    V_equalized = global_histogram_equalization(V_channel)

    #put equalized V back into HSV image (normalized to [0, 1])
    hsv[:, :, 2] = V_equalized.astype(np.float32) / 255.0

    #convert HSV back to RGB using our from-scratch function
    img_rgb_eq = hsv_to_rgb(hsv)

    #convert RGB back to BGR for OpenCV compatibility
    return cv2.cvtColor(img_rgb_eq, cv2.COLOR_RGB2BGR)


def equalize_color_image_rgb_channels(img_bgr):

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # convert to RGB
    result = np.zeros_like(img_rgb)

    for i in range(3):  # iterate over R, G, B channels
        result[:, :, i] = global_histogram_equalization(img_rgb[:, :, i])

    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)  # back to BGR


#VISUALIZATION
def compare_grayscale_results(original, equalized, method_name="GHE", save_path=None):

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    #Original image
    axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    #Original histogram
    hist_orig, cdf_orig = compute_histogram_and_cdf(original)
    axes[0, 1].bar(range(256), hist_orig, color='gray', width=1.0)
    axes[0, 1].set_title("Original Histogram")
    axes[0, 1].set_xlabel("Intensity")
    axes[0, 1].set_ylabel("Frequency")

    #Original CDF
    axes[0, 2].plot(cdf_orig, color='blue')
    axes[0, 2].set_title("Original CDF")
    axes[0, 2].set_xlabel("Intensity")
    axes[0, 2].set_ylabel("CDF")
    axes[0, 2].set_ylim([0, 1])

    #Equalized image
    axes[1, 0].imshow(equalized, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f"Equalized Image ({method_name})")
    axes[1, 0].axis('off')

    #Equalized histogram
    hist_eq, cdf_eq = compute_histogram_and_cdf(equalized)
    axes[1, 1].bar(range(256), hist_eq, color='gray', width=1.0)
    axes[1, 1].set_title(f"Equalized Histogram ({method_name})")
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Frequency")

    #Equalized CDF
    axes[1, 2].plot(cdf_eq, color='red')
    axes[1, 2].set_title(f"Equalized CDF ({method_name})")
    axes[1, 2].set_xlabel("Intensity")
    axes[1, 2].set_ylabel("CDF")
    axes[1, 2].set_ylim([0, 1])

    plt.suptitle(f"Histogram Equalization — {method_name}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def compare_color_results(original_bgr, hsv_eq_bgr, rgb_eq_bgr, save_path=None):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    #convert BGR to RGB for correct display in matplotlib
    axes[0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(hsv_eq_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title("HE on V channel (HSV)\nPreserves color")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(rgb_eq_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title("HE on R,G,B independently\nColor distortion")
    axes[2].axis('off')

    plt.suptitle("Color Image Histogram Equalization Comparison", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
