import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def plot_transformation(r, s, original, enhanced, title, save_path=None):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(r, s, color='blue')
    axes[0].set_title(f'{title} Curve: $s = T(r)$')
    axes[0].set_xlabel('Input Intensity ($r$)')
    axes[0].set_ylabel('Output Intensity ($s$)')
    axes[0].grid(True)
    
    if len(original.shape) == 3:
        axes[1].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[1].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Original Image')
    axes[1].axis('off')
    
    if len(enhanced.shape) == 3:
        axes[2].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    else:
        axes[2].imshow(enhanced, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Enhanced Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {save_path}")


#Image Negative
def image_negative(img):
    
    # s = L - 1 - r, where L = 256 for 8-bit images
    L = 256                                          #intensity range: 0_255
    enhanced = (L - 1) - img                         #apply the negative transformation
    
    r_vals = np.arange(0, 256)                       #all inputs, 0 to 255
    s_vals = (L - 1) - r_vals                        #output of the negative transform for each r
    
    plot_transformation(r_vals, s_vals, img, enhanced, "Image Negative")
    return enhanced    

#Logarithmic Transformation
def log_transform(img):
    
    c = 255 / np.log(1 + np.max(img))                    #normalize to 0-255 safely
    enhanced = c * (np.log(1 + img.astype(np.float32)))  #convert image to float
    enhanced = np.array(enhanced, dtype=np.uint8)        #convert back to unit8
    
    r_vals = np.arange(0, 256)
    s_vals = c * np.log(1 + r_vals)
    
    plot_transformation(r_vals, s_vals, img, enhanced, "Logarithmic")
    return enhanced

#Gamma Correction
def gamma_correction(img, gamma):                       #img: input image, gamma: chosen gamma value
    
    # s = c * r^gamma
    r_norm = img.astype(np.float32) / 255.0             #normalize r to [0, 1] for applying gamma, then scale back to [0, 255]
    s_norm = np.power(r_norm, gamma)                    #apply gamma correction
    enhanced = np.uint8(s_norm * 255)                   #multiply by 255 to bring back to 8-bit range/ convert to unit8
    
    r_vals = np.arange(0, 256)
    s_vals = 255 * np.power(r_vals / 255.0, gamma)
    
    plot_transformation(r_vals, s_vals, img, enhanced, f"Gamma ($\\gamma={gamma}$)")
    return enhanced

#Piecewise-Linear Transformation
def piecewise_linear(img, r1, s1, r2, s2):                           #regions: 0 to 0, r1 to s1, r2 to s2, 255 to 255
    def pixel_val(pix, r1, s1, r2, s2):                             #output intensity for a single pixel
        if (0 <= pix and pix <= r1):
            if r1 == 0:
                return 0
            return (s1 / r1) * pix                                  #controls how the darkest intensities are stretched
        elif (r1 < pix and pix <= r2):
            return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1        #where contrast stretching happens
        else:
            return ((255 - s2) / (255 - r2)) * (pix - r2) + s2      #controls how the brightest parts are enhanced

    pixel_val_vec = np.vectorize(pixel_val)                         
    enhanced = pixel_val_vec(img, r1, s1, r2, s2).astype(np.uint8)  ##generates the full piecewise‑transformed image
    
    r_vals = np.arange(0, 256)
    s_vals = pixel_val_vec(r_vals, r1, s1, r2, s2)
    
    plot_transformation(r_vals, s_vals, img, enhanced, "Piecewise-Linear")
    return enhanced

def mse(imageA, imageB):
    #Calculate the Mean Squared Error between two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def bit_plane_slicing(img, save_path=None):
    #Extract 8 individual bit-planes
    bit_planes = []
    for i in range(8):
        # Extract the i-th bit and scale to 0-255 for visibility
        plane = (img & (1 << i)) >> i
        bit_planes.append(plane * 255)
        
    # Plot the 8 bit-planes
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    fig.suptitle('8 Bit-Planes')
    for i in range(8):
        ax = axes[i // 4, i % 4]
        ax.imshow(bit_planes[i], cmap='gray')
        ax.set_title(f'Bit-Plane {i} (0=LSB)')
        ax.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path.replace('.png', '_bitplanes.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Reconstruct using top 4 MSBs (Bits 4, 5, 6, 7)
    msb_reconstruction = (img & 240) # 240 is 11110000 in binary
    
    # Reconstruct using 4 LSBs (Bits 0, 1, 2, 3)
    lsb_reconstruction = (img & 15)  # Keep raw for return
    lsb_reconstruction_display = lsb_reconstruction * 17  # Scale for display

    # Calculate MSE
    mse_msb = mse(img, msb_reconstruction)
    mse_lsb = mse(img, lsb_reconstruction_display)

    # Plot original and reconstructions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(msb_reconstruction, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Top 4 MSBs\nMSE: {mse_msb:.2f}')
    axes[1].axis('off')
    
    axes[2].imshow(lsb_reconstruction_display, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'Bottom 4 LSBs\nMSE: {mse_lsb:.2f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_reconstruction.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return msb_reconstruction, lsb_reconstruction



