# Digital Image Processing Fundamentals

## Overview

This project presents a collection of fundamental Digital Image Processing techniques implemented as part of the Digital Image Processing course at Shiraz University.

The project is organized into several sections, each focusing on a different concept in spatial-domain image processing.

The implementations cover:

- Matrix and coordinate transformations
- Image downsampling
- Image upsampling and interpolation
- Nearest-neighbor interpolation
- Bilinear interpolation
- Bicubic interpolation
- Image reconstruction quality assessment
- Point operations
- Gamma correction
- Piecewise-linear transformations
- Image negative transformation
- Global Histogram Equalization
- Local Histogram Equalization
- RGB to HSV conversion
- HSV to RGB conversion
- Color-image contrast enhancement
- Histogram Specification / Histogram Matching

Several algorithms were implemented manually rather than relying on high-level image-processing functions.

---

# Project Sections

The project is divided into four main parts.

## Section 0 - Warm-Up

The warm-up section focuses on matrix manipulation and coordinate transformation.

A smaller matrix is placed inside a larger matrix using a diamond-shaped transformation.

For a small matrix element located at `(i, j)`, the transformed coordinates are calculated using:

```text
row = cx + (i - j)

col = cy + (i + j - (m - 1))
```

where:

- `cx` and `cy` represent the center of the larger matrix
- `i` and `j` represent coordinates in the smaller matrix
- `m` is the number of columns in the smaller matrix

This section provides practice with coordinate mapping before moving to image-based operations.

---

# Section 1 - Pixel Resolution and Image Interpolation

## Overview

This section investigates spatial image resampling.

Two main operations are studied:

1. Downsampling
2. Upsampling / Reconstruction

The objective is to understand how reducing image resolution affects image information and how different interpolation techniques reconstruct the original resolution.

---

## Downsampling

The image is downsampled using the following factors:

```text
N = 2
N = 4
N = 8
```

Direct pixel sampling is used.

Conceptually:

```python
downsampled = image[::N, ::N]
```

This keeps every `N`-th pixel in both spatial dimensions.

The method is computationally inexpensive but may remove important image details and introduce aliasing.

---

## Image Reconstruction

After downsampling, the reduced image is resized back to approximately its original resolution.

Three interpolation methods are implemented:

- Nearest Neighbor
- Bilinear Interpolation
- Bicubic Interpolation

---

## Nearest-Neighbor Interpolation

Nearest-neighbor interpolation assigns each new pixel the value of the closest pixel in the source image.

The method is:

- Very fast
- Simple to implement
- Able to preserve hard edges

However, it often creates:

- Blocky artifacts
- Staircase effects
- Jagged diagonal edges

---

## Bilinear Interpolation

Bilinear interpolation estimates a new pixel using four neighboring pixels.

The value is computed as a weighted combination of the surrounding pixels.

Compared with nearest-neighbor interpolation, it generally produces:

- Smoother images
- Fewer block artifacts
- Better visual continuity

However, some image sharpness may be lost.

---

## Bicubic Interpolation

Bicubic interpolation uses a larger neighborhood of 16 pixels.

A cubic interpolation kernel is applied to estimate each new pixel.

Compared with nearest-neighbor and bilinear interpolation, bicubic interpolation generally produces smoother and visually higher-quality reconstructions.

The increased quality comes at the cost of greater computational complexity.

---

# Reconstruction Evaluation

The reconstructed images are evaluated both visually and quantitatively.

Two common image-quality metrics are used.

## Mean Squared Error

Mean Squared Error measures the average squared difference between the original image and the reconstructed image.

```text
MSE = mean((I_original - I_reconstructed)^2)
```

Lower MSE indicates better reconstruction.

---

## Peak Signal-to-Noise Ratio

PSNR measures reconstruction quality relative to the maximum possible pixel intensity.

```text
PSNR = 10 log10(MAX^2 / MSE)
```

Higher PSNR indicates better reconstruction quality.

---

## Interpolation Comparison

The three interpolation methods demonstrate a trade-off between computational cost and visual quality.

| Method | Speed | Smoothness | Typical Quality |
| --- | --- | --- | --- |
| Nearest Neighbor | High | Low | Basic |
| Bilinear | Medium | Medium | Good |
| Bicubic | Lower | High | Usually Best |

The effect becomes more visible when larger downsampling factors such as `N = 8` are used.

---

# Section 2 - Point Operations

## Overview

Point operations modify each pixel independently according to a transformation function.

The experiments investigate several techniques for adjusting image brightness and contrast.

---

## Image Negative

The negative transformation reverses intensity values.

For an 8-bit grayscale image:

```text
s = 255 - r
```

where:

- `r` is the original intensity
- `s` is the transformed intensity

Negative transformation can make certain structures easier to observe, particularly in grayscale and medical images.

---

## Gamma Correction

Gamma correction applies a nonlinear intensity transformation.

Conceptually:

```text
s = c * r^gamma
```

Changing `gamma` affects different regions of the intensity range.

Low and high gamma values can be used to modify dark or bright image regions differently.

Gamma correction is useful for brightness adjustment but does not always provide enough control for images with complicated intensity distributions.

---

## Piecewise-Linear Transformation

Piecewise-linear contrast transformation provides more direct control over different intensity intervals.

Instead of applying a single transformation to the full range, several linear segments are used.

This allows selected ranges of intensities to be:

- Expanded
- Compressed
- Preserved

The experiments showed that piecewise-linear transformations can reveal structural details more effectively than a simple global brightness adjustment in some images.

---

# Section 3 - Histogram Processing

## Global Histogram Equalization

Global Histogram Equalization attempts to improve contrast by redistributing intensity values.

For an image with dimensions:

```text
M x N
```

the probability of intensity `r_k` is:

```text
P(r_k) = n_k / (M * N)
```

where `n_k` is the number of pixels with intensity `r_k`.

The cumulative distribution function is then calculated:

```text
CDF(r_k) = sum(P(r_j))
```

The new intensity value is obtained using:

```text
s_k = round((L - 1) * CDF(r_k))
```

where:

```text
L = 256
```

for an 8-bit image.

---

## Global Histogram Equalization Analysis

Global Histogram Equalization can improve the overall contrast of an image by using a larger portion of the available intensity range.

However, because one transformation is used for the entire image, small local structures may remain difficult to observe.

---

# Local Histogram Equalization

Local Histogram Equalization calculates a different histogram transformation for each local image region.

A sliding neighborhood such as:

```text
15 x 15
```

is placed around each pixel.

A histogram and CDF are computed using pixels inside that neighborhood, and the center pixel is transformed using the local distribution.

This allows contrast enhancement to adapt to different image regions.

---

## Global vs. Local Histogram Equalization

The experiments demonstrate an important trade-off.

Global HE is computationally efficient but may fail to reveal small local structures.

Local HE can reveal considerably more local detail, but it can also:

- Amplify noise
- Produce strong texture artifacts
- Require much more computation

In the recorded experiment:

```text
Global Histogram Equalization ≈ 0.15 seconds

Local Histogram Equalization ≈ 43.65 seconds
```

Therefore, the local method required roughly:

```text
290x
```

more execution time.

---

# Color Image Enhancement

Histogram operations become more complicated for RGB images.

Applying Histogram Equalization independently to:

```text
R
G
B
```

can alter the relative relationships between the three channels.

This can produce:

- Hue shifts
- Unnatural colors
- Color imbalance

To reduce this problem, the image is transformed into the HSV color space.

---

# RGB to HSV Conversion

The RGB-to-HSV transformation is implemented mathematically.

First, the RGB values are normalized.

Then:

```text
Cmax = max(R, G, B)

Cmin = min(R, G, B)

Delta = Cmax - Cmin
```

The Value component is:

```text
V = Cmax
```

The Saturation component depends on `Delta` and `Cmax`, while Hue is calculated according to which RGB component is maximal.

---

# Histogram Equalization in HSV

Instead of equalizing the three RGB channels separately, Histogram Equalization is applied only to:

```text
V
```

The Hue and Saturation channels remain unchanged.

```text
V_enhanced = HE(V)
```

The enhanced HSV image is then converted back to RGB.

This approach improves brightness and contrast while preserving the original color relationships more effectively.

---

# Section 4 - Histogram Specification

## Overview

Histogram Specification, also called Histogram Matching, transforms the intensity distribution of a source image so that it resembles the intensity distribution of a reference image.

Unlike Histogram Equalization, the desired distribution is not necessarily uniform.

The general relationship is:

```text
z = G^-1(T(r))
```

where:

- `T(r)` is the normalized CDF of the source image
- `G(z)` is the normalized CDF of the reference image
- `G^-1` represents the inverse mapping

---

# Grayscale Conversion

Before histogram matching, RGB images are converted manually to grayscale.

The standard luminance equation is used:

```text
Y = 0.299R + 0.587G + 0.114B
```

This conversion is implemented directly rather than using functions such as:

```text
cv2.cvtColor()
```

---

# Histogram Computation

For each possible grayscale intensity:

```text
r = 0, 1, ..., 255
```

the histogram counts the number of pixels having that intensity.

Conceptually:

```text
h(r) = number of pixels with intensity r
```

---

# Normalized CDF

The histogram is converted into a cumulative distribution.

Normalization ensures:

```text
0 <= CDF(r) <= 1
```

and:

```text
CDF(255) = 1
```

Normalized CDFs make it possible to compare images containing different numbers of pixels.

---

# Histogram Matching Algorithm

The histogram-matching procedure consists of four main stages.

### Step 1

Calculate the normalized source CDF:

```text
T(r)
```

### Step 2

Calculate the normalized reference CDF:

```text
G(z)
```

### Step 3

For each source intensity `r`, find the reference intensity `z` whose CDF is closest:

```text
z = argmin |G(z) - T(r)|
```

### Step 4

Store these values in a lookup table and transform all source pixels:

```text
I_output(i,j) = mapping[I_source(i,j)]
```

This produces an output image whose intensity distribution more closely resembles that of the reference image.

---

# Discrete Mapping

In digital images, the CDF is discrete and therefore does not usually have an exact mathematical inverse.

For this reason, the implementation approximates:

```text
G^-1(T(r))
```

by searching for the reference intensity whose CDF has the smallest absolute difference from the source CDF.

This behavior is expected in discrete Histogram Specification.

---

# Project Structure

```text
digital-image-processing-fundamentals/
│
├── README.md
│
├── Section0_WarmUp/
│   ├── Code.py
│   └── Report.pdf
│
├── Section1_Pixel_Resolution/
│   ├── section1_pixel_resolution.py
│   ├── input.png
│   ├── output_section1/
│   └── Report.pdf
│
├── Section2_3_Histogram_Processing/
│   ├── HE.py
│   ├── HE-test.py
│   ├── PO.py
│   ├── PO-test.py
│   └── Report-en.pdf
│
└── Section4_Histogram_Specification/
    ├── Dip.ipynb
    └── Report.pdf
```

---

# File Description

| File / Directory | Description |
| --- | --- |
| `Section0_WarmUp/Code.py` | Matrix coordinate transformation and diamond placement |
| `section1_pixel_resolution.py` | Downsampling, interpolation, reconstruction, and quality evaluation |
| `Section1_Pixel_Resolution/output_section1/` | Generated downsampled and reconstructed images |
| `HE.py` | Histogram-based image enhancement implementations |
| `HE-test.py` | Tests and experiments for histogram processing |
| `PO.py` | Point-operation implementations |
| `PO-test.py` | Experiments for point transformations |
| `Dip.ipynb` | Histogram Specification and related image-processing experiments |
| Report files | Detailed documentation and analysis for individual sections |

---

# Main Topics Covered

- Digital Image Processing
- Spatial-Domain Processing
- Image Resampling
- Downsampling
- Upsampling
- Image Interpolation
- Nearest-Neighbor Interpolation
- Bilinear Interpolation
- Bicubic Interpolation
- MSE
- PSNR
- Point Operations
- Image Negative
- Gamma Correction
- Contrast Stretching
- Histograms
- Global Histogram Equalization
- Local Histogram Equalization
- RGB and HSV Color Spaces
- Color Image Enhancement
- Histogram Specification
- Histogram Matching
- Cumulative Distribution Functions

---

# Key Observations

The experiments demonstrate several practical characteristics of classical image-processing algorithms:

- Strong downsampling removes fine spatial information.
- Nearest-neighbor interpolation is fast but produces visible block artifacts.
- Bilinear interpolation provides smoother reconstruction.
- Bicubic interpolation generally provides better visual quality at greater computational cost.
- Global Histogram Equalization efficiently improves overall contrast but may miss local detail.
- Local Histogram Equalization enhances small structures but significantly increases computation and may amplify noise.
- Equalizing RGB channels independently can distort image colors.
- Performing intensity enhancement in HSV space can preserve hue and saturation more effectively.
- Histogram Matching allows the intensity characteristics of one image to be transferred to another image.

---

# Technologies

- Python
- NumPy
- Matplotlib
- Jupyter Notebook
- Digital Image Processing
- Image Interpolation
- Histogram Processing
- Color Space Transformation

---

# Course Information

**Course:** Digital Image Processing  
**Instructor:** Prof. Zohreh Azimifar  
**Department:** Computer Science and Engineering  
**University:** Shiraz University  
**Year:** 2026

---

# Authors

Saghar Kheradmand  
Samira Khandan  
AmirHossein Jeddi
