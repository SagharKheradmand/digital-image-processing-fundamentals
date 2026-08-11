import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))

#helpers

def load_gray(path: str) -> np.ndarray:
    #load an image
    try:
        img = Image.open(path).convert("L")  # Convert to grayscale
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not read image: {path}")
    
    return np.array(img, dtype=np.float64)


def save_gray(img: np.ndarray, path: str) -> None:
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)

def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Squared Error between two arrays."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    #Peak Signal-to-Noise Ratio (dB).
    err = mse(a, b)
    if err == 0:
        return float('inf')
    return 10.0 * np.log10(max_val ** 2 / err)


# 1. Downsampling

def downsample(img: np.ndarray, N: int) -> np.ndarray:
    #Reduce resolution by factor N via direct pixel sampling.
    #Output shape: (H//N, W//N)
    H, W = img.shape
    return img[::N, ::N].copy()


# 2a. Nearest Neighbor Upsampling

def upsample_nearest(img_down: np.ndarray, N: int,
                     target_H: int, target_W: int) -> np.ndarray:
    """
    Nearest-neighbor interpolation.

    For each target pixel (x, y), the source coordinate in the
    downsampled image is simply round(x/N), round(y/N), clamped to bounds.
    """
    H_d, W_d = img_down.shape
    out = np.zeros((target_H, target_W), dtype=np.float64)

    for x in range(target_H):
        for y in range(target_W):
            # Map target → source
            src_i = int(round(x / N))
            src_j = int(round(y / N))
            # Clamp
            src_i = min(src_i, H_d - 1)
            src_j = min(src_j, W_d - 1)
            out[x, y] = img_down[src_i, src_j]

    return out


def upsample_nearest_fast(img_down: np.ndarray, N: int,
                           target_H: int, target_W: int) -> np.ndarray:
    """Vectorised nearest-neighbor (same math, ~100× faster)."""
    H_d, W_d = img_down.shape

    xs = np.arange(target_H)
    ys = np.arange(target_W)

    src_i = np.clip(np.round(xs / N).astype(int), 0, H_d - 1)
    src_j = np.clip(np.round(ys / N).astype(int), 0, W_d - 1)

    # Broadcast indexing
    return img_down[np.ix_(src_i, src_j)].copy()


# 2b. Bilinear Upsampling

def K_bilin(t: np.ndarray) -> np.ndarray:
    """
    Bilinear (tent) kernel:
        K(t) = 1 - |t|   if |t| <= 1
               0          otherwise
    """
    t = np.abs(t)
    return np.where(t <= 1.0, 1.0 - t, 0.0)


def upsample_bilinear(img_down: np.ndarray, N: int,
                      target_H: int, target_W: int) -> np.ndarray:
    """
    Bilinear interpolation as specified in the homework:

        I_up(x, y) = sum_{m=0}^{1} sum_{n=0}^{1}
                         I_down(i0+m, j0+n)
                         * K_bilin(x/N - (i0+m))
                         * K_bilin(y/N - (j0+n))

    where i0 = floor(x/N), j0 = floor(y/N).
    """
    H_d, W_d = img_down.shape
    out = np.zeros((target_H, target_W), dtype=np.float64)

    xs = np.arange(target_H)
    ys = np.arange(target_W)

    # Continuous source coordinates
    fx = xs / N          # shape (target_H,)
    fy = ys / N          # shape (target_W,)

    i0 = np.floor(fx).astype(int)   # (target_H,)
    j0 = np.floor(fy).astype(int)   # (target_W,)

    for m in range(2):        # 0, 1
        for n in range(2):    # 0, 1
            ii = np.clip(i0 + m, 0, H_d - 1)   # (target_H,)
            jj = np.clip(j0 + n, 0, W_d - 1)   # (target_W,)

            # Kernel weights
            wx = K_bilin(fx - (i0 + m))         # (target_H,)
            wy = K_bilin(fy - (j0 + n))         # (target_W,)

            # Outer product of weights × pixel values
            # img_down[ii, :][:, jj] is (target_H, target_W)
            pixel_vals = img_down[np.ix_(ii, jj)]
            out += pixel_vals * np.outer(wx, wy)

    return out


# 2c. Bicubic Upsampling

def K_cubic(t: np.ndarray) -> np.ndarray:
    """
    Keys' cubic convolution kernel:
        K(t) = 1.5|t|^3 - 2.5|t|^2 + 1       if |t| <= 1
               -0.5|t|^3 + 2.5|t|^2 - 4|t| + 2  if 1 < |t| < 2
               0                                  if |t| >= 2
    """
    t = np.abs(t)
    result = np.zeros_like(t)

    mask1 = t <= 1.0
    result[mask1] = (1.5 * t[mask1]**3
                     - 2.5 * t[mask1]**2
                     + 1.0)

    mask2 = (t > 1.0) & (t < 2.0)
    result[mask2] = (-0.5 * t[mask2]**3
                     + 2.5 * t[mask2]**2
                     - 4.0 * t[mask2]
                     + 2.0)

    return result


def upsample_bicubic(img_down: np.ndarray, N: int,
                     target_H: int, target_W: int) -> np.ndarray:
    """
    Bicubic interpolation:

        I_up(x, y) = sum_{m=-1}^{2} sum_{n=-1}^{2}
                         I_down(i0+m, j0+n)
                         * K_cubic(dx - m)
                         * K_cubic(dy - n)

    where i0 = floor(x/N), dx = x/N - i0,
          j0 = floor(y/N), dy = y/N - j0.
    """
    H_d, W_d = img_down.shape
    out = np.zeros((target_H, target_W), dtype=np.float64)

    fx = np.arange(target_H) / N     # (target_H,)
    fy = np.arange(target_W)  / N    # (target_W,)

    i0 = np.floor(fx).astype(int)    # (target_H,)
    j0 = np.floor(fy).astype(int)    # (target_W,)

    dx = fx - i0                      # (target_H,)
    dy = fy - j0                      # (target_W,)

    for m in range(-1, 3):    # -1, 0, 1, 2
        for n in range(-1, 3):
            ii = np.clip(i0 + m, 0, H_d - 1)
            jj = np.clip(j0 + n, 0, W_d - 1)

            wx = K_cubic(dx - m)      # (target_H,)
            wy = K_cubic(dy - n)      # (target_W,)

            pixel_vals = img_down[np.ix_(ii, jj)]
            out += pixel_vals * np.outer(wx, wy)

    # Clamp to valid range
    return np.clip(out, 0.0, 255.0)


# 3. Visualisation helpers

def show_zoomed_patch(ax, img: np.ndarray, title: str,
                      patch_r: tuple = (100, 150),
                      patch_c: tuple = (100, 150),
                      cmap: str = 'gray'):
    """Display image with a red rectangle and inset zoom."""
    ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis('off')

    r0, r1 = patch_r
    c0, c1 = patch_c
    rect = patches.Rectangle((c0, r0), c1 - c0, r1 - r0,
                              linewidth=1.5, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # inset axes
    axins = ax.inset_axes([0.55, 0.0, 0.45, 0.45])
    axins.imshow(img[r0:r1, c0:c1], cmap=cmap, vmin=0, vmax=255,
                 interpolation='nearest')
    axins.set_xticks([])
    axins.set_yticks([])
    axins.spines['bottom'].set_color('red')
    axins.spines['top'].set_color('red')
    axins.spines['left'].set_color('red')
    axins.spines['right'].set_color('red')


def plot_results(original: np.ndarray, N: int,
                 results: dict,
                 metrics: dict,
                 save_path: str = None):
    
    #ide-by-side comparison of all three interpolation methods
    
    methods = ['Nearest Neighbor', 'Bilinear', 'Bicubic']
    keys    = ['nearest', 'bilinear', 'bicubic']

    H, W = original.shape
    # Choose a sensible patch (avoid border)
    pr = (H // 4, H // 4 + H // 8)
    pc = (W // 4, W // 4 + W // 8)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f'Section 1.1 — Downsampling factor N={N}', fontsize=13)

    show_zoomed_patch(axes[0], original,
                      f'Original\n({H}×{W})', pr, pc)

    for ax, k, name in zip(axes[1:], keys, methods):
        img_up = results[k]
        m = metrics[k]
        show_zoomed_patch(ax, img_up,
                          f'{name}\nMSE={m["mse"]:.2f}  PSNR={m["psnr"]:.2f} dB',
                          pr, pc)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# 4. Main pipeline

def run_section1(image_path: str, output_dir: str = 'outputs_section1'):
    output_dir = os.path.join(script_dir, 'output_section1')

    original = load_gray(image_path)
    H, W = original.shape
    print(f"Loaded image: {image_path}  ({H}×{W})")

    factors = [2, 4, 8]
    all_metrics = {}

    for N in factors:
        print(f"\n── Downsampling factor N = {N} ──")

        # ---- Downsample ----
        img_down = downsample(original, N)
        Hd, Wd = img_down.shape
        print(f"  Downsampled: {Hd}×{Wd}")
        save_gray(img_down,
                  os.path.join(output_dir, f'downsampled_N{N}.png'))

        #  Upsample 
        up_nn  = upsample_nearest_fast(img_down, N, H, W)
        up_bil = upsample_bilinear(img_down, N, H, W)
        up_bic = upsample_bicubic(img_down, N, H, W)

        #  Metrics
        def compute_metrics(img_up):
            return {
                'mse':  mse(original, img_up),
                'psnr': psnr(original, img_up),
            }

        metrics = {
            'nearest':  compute_metrics(up_nn),
            'bilinear': compute_metrics(up_bil),
            'bicubic':  compute_metrics(up_bic),
        }
        all_metrics[N] = metrics

        for name, m in metrics.items():
            print(f"  {name:12s}  MSE={m['mse']:8.3f}  PSNR={m['psnr']:.3f} dB")

        #  Save upsampled images 
        for k, img_up in [('nearest', up_nn), ('bilinear', up_bil),
                           ('bicubic', up_bic)]:
            save_gray(img_up,
                      os.path.join(output_dir, f'upsampled_{k}_N{N}.png'))

        #  Plot 
        plot_results(
            original, N,
            {'nearest': up_nn, 'bilinear': up_bil, 'bicubic': up_bic},
            metrics,
            save_path=os.path.join(output_dir, f'comparison_N{N}.png')
        )

    #  Summary table 
    print("\n\n═══ METRIC SUMMARY ═══")
    header = f"{'N':>3}  {'Method':15}  {'MSE':>10}  {'PSNR (dB)':>10}"
    print(header)
    print('─' * len(header))
    for N in factors:
        for name, m in all_metrics[N].items():
            print(f"{N:>3}  {name:15}  {m['mse']:10.3f}  {m['psnr']:10.3f}")
        print()

    return all_metrics


# Entry point

if __name__ == '__main__':
    
    if os.path.isfile(os.path.join(script_dir, "input.png")):
        img_name = "input.png"
        img_path = os.path.join(script_dir, img_name)
    else:
        sample = np.zeros((512, 512), dtype=np.uint8)

        for i in range(512):
            for j in range(512):
                sample[i, j] = int(
                    128
                    + 60 * np.sin(2 * np.pi * i / 64)
                    + 40 * np.cos(2 * np.pi * j / 32)
                )

        img_path = 'sample_image.png'
        Image.fromarray(sample).save(img_path)
        print(f"No image path provided — using generated sample: {img_path}")
        


    run_section1(img_path)
    print("===========================================")
    print("All results saved on output_section1 Folder")
    print("===========================================")
    print("If you want to run the code with another picture, convert its name and format like this(input.png) and place it in the same place with .py file")
    print("==========================================================================================================================================")
