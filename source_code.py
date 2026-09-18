# ============================================================
# SmartFruit Vision
# Explainable Fruit Surface Damage Analysis Using
# Digital Image Processing
#
# Student: Aneena Binoy
# Register No: 24UBC110
# Course: BCA Honours
# Institution: Marian College Kuttikkanam
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from skimage import exposure
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from sklearn.cluster import KMeans


# ------------------------------------------------------------
# 2. DATASET PATH
# ------------------------------------------------------------

train_path = "/content/drive/MyDrive/SmartFruit_Vision/train"

categories = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]


# ------------------------------------------------------------
# 3. CHECK DATASET
# ------------------------------------------------------------

print("Checking dataset...\n")

total = 0

for category in categories:

    folder_path = os.path.join(train_path, category)

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    print(f"{category}: {len(images)} images")

    total += len(images)

print("\nTotal images:", total)


# ------------------------------------------------------------
# 4. DISPLAY DATASET SAMPLES
# ------------------------------------------------------------

samples = [
    ("freshapples", "Fresh Apple"),
    ("rottenapples", "Rotten Apple"),
    ("freshbanana", "Fresh Banana"),
    ("rottenbanana", "Rotten Banana"),
    ("freshoranges", "Fresh Orange"),
    ("rottenoranges", "Rotten Orange")
]

plt.figure(figsize=(12, 8))

for i, (folder, title) in enumerate(samples):

    folder_path = os.path.join(train_path, folder)

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    image_path = os.path.join(folder_path, files[0])

    image = Image.open(image_path)

    plt.subplot(2, 3, i + 1)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")

plt.suptitle("Dataset Samples")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 5. LOAD ROTTEN APPLE 20
# ------------------------------------------------------------

folder_path = os.path.join(train_path, "rottenapples")

rotten_files = [
    f for f in os.listdir(folder_path)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]

rotten_path = os.path.join(
    folder_path,
    rotten_files[19]
)

rotten_image = np.array(
    Image.open(rotten_path).convert("RGB")
)


# ------------------------------------------------------------
# 6. ADD GAUSSIAN NOISE
# ------------------------------------------------------------

noise = np.random.normal(
    0,
    25,
    rotten_image.shape
)

noisy_image = (
    rotten_image.astype(np.float32) + noise
)

noisy_image = np.clip(
    noisy_image,
    0,
    255
).astype(np.uint8)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(rotten_image)
plt.title("Original Rotten Apple 20")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(noisy_image)
plt.title("Image with Gaussian Noise")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 7. IMAGE RESTORATION USING FILTERS
# ------------------------------------------------------------

noisy_bgr = cv2.cvtColor(
    noisy_image,
    cv2.COLOR_RGB2BGR
)

mean_filtered = cv2.blur(
    noisy_bgr,
    (5, 5)
)

median_filtered = cv2.medianBlur(
    noisy_bgr,
    5
)

gaussian_filtered = cv2.GaussianBlur(
    noisy_bgr,
    (5, 5),
    0
)

mean_filtered = cv2.cvtColor(
    mean_filtered,
    cv2.COLOR_BGR2RGB
)

median_filtered = cv2.cvtColor(
    median_filtered,
    cv2.COLOR_BGR2RGB
)

gaussian_filtered = cv2.cvtColor(
    gaussian_filtered,
    cv2.COLOR_BGR2RGB
)


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(noisy_image)
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(mean_filtered)
plt.title("Mean Filter")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(median_filtered)
plt.title("Median Filter")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(gaussian_filtered)
plt.title("Gaussian Filter")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8. RESTORATION EVALUATION
# ------------------------------------------------------------

mse_mean = mean_squared_error(
    rotten_image,
    mean_filtered
)

mse_median = mean_squared_error(
    rotten_image,
    median_filtered
)

mse_gaussian = mean_squared_error(
    rotten_image,
    gaussian_filtered
)


psnr_mean = peak_signal_noise_ratio(
    rotten_image,
    mean_filtered
)

psnr_median = peak_signal_noise_ratio(
    rotten_image,
    median_filtered
)

psnr_gaussian = peak_signal_noise_ratio(
    rotten_image,
    gaussian_filtered
)


ssim_mean = structural_similarity(
    rotten_image,
    mean_filtered,
    channel_axis=2
)

ssim_median = structural_similarity(
    rotten_image,
    median_filtered,
    channel_axis=2
)

ssim_gaussian = structural_similarity(
    rotten_image,
    gaussian_filtered,
    channel_axis=2
)


print("Restoration Evaluation")
print("----------------------")

print("\nMean Filter")
print("MSE :", mse_mean)
print("PSNR:", psnr_mean)
print("SSIM:", ssim_mean)

print("\nMedian Filter")
print("MSE :", mse_median)
print("PSNR:", psnr_median)
print("SSIM:", ssim_median)

print("\nGaussian Filter")
print("MSE :", mse_gaussian)
print("PSNR:", psnr_gaussian)
print("SSIM:", ssim_gaussian)


# ------------------------------------------------------------
# 9. HISTOGRAM ENHANCEMENT
# ------------------------------------------------------------

restored_image = gaussian_filtered

gray_image = cv2.cvtColor(
    restored_image,
    cv2.COLOR_RGB2GRAY
)

hist_equalized = exposure.equalize_hist(
    gray_image
)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_image = clahe.apply(
    gray_image
)


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(hist_equalized, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(clahe_image, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10. PSNR COMPARISON
# ------------------------------------------------------------

methods = [
    "Mean",
    "Median",
    "Gaussian"
]

psnr_values = [
    psnr_mean,
    psnr_median,
    psnr_gaussian
]

plt.figure(figsize=(7, 5))

plt.bar(
    methods,
    psnr_values
)

plt.title("PSNR Comparison")
plt.xlabel("Filtering Method")
plt.ylabel("PSNR (dB)")

plt.show()


# ------------------------------------------------------------
# 11. LOAD FRESH APPLE 7 AND ROTTEN APPLE 20
# ------------------------------------------------------------

fresh_folder = os.path.join(
    train_path,
    "freshapples"
)

fresh_files = [
    f for f in os.listdir(fresh_folder)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]

fresh_path = os.path.join(
    fresh_folder,
    fresh_files[7]
)

fresh_image = np.array(
    Image.open(fresh_path).convert("RGB")
)

print("Fresh Apple 7:", fresh_path)
print("Rotten Apple 20:", rotten_path)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(fresh_image)
plt.title("Fresh Apple 7")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(rotten_image)
plt.title("Rotten Apple 20")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 12. FRUIT SEGMENTATION USING GRABCUT
# ------------------------------------------------------------

def get_fruit_mask_grabcut(image):

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    height, width = image.shape[:2]

    margin_x = int(width * 0.05)
    margin_y = int(height * 0.05)

    rectangle = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y
    )

    mask = np.zeros(
        (height, width),
        np.uint8
    )

    background_model = np.zeros(
        (1, 65),
        np.float64
    )

    foreground_model = np.zeros(
        (1, 65),
        np.float64
    )

    cv2.grabCut(
        image_bgr,
        mask,
        rectangle,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    fruit_mask = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    kernel = np.ones(
        (7, 7),
        np.uint8
    )

    fruit_mask = cv2.morphologyEx(
        fruit_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    fruit_mask = cv2.morphologyEx(
        fruit_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        fruit_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        clean_mask = np.zeros_like(
            fruit_mask
        )

        cv2.drawContours(
            clean_mask,
            [largest_contour],
            -1,
            255,
            -1
        )

        fruit_mask = clean_mask

    return fruit_mask


fresh_mask = get_fruit_mask_grabcut(
    fresh_image
)

rotten_mask = get_fruit_mask_grabcut(
    rotten_image
)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(fresh_mask, cmap="gray")
plt.title("Fresh Apple Mask")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(rotten_mask, cmap="gray")
plt.title("Rotten Apple Mask")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 13. EXTRACT FRUIT FROM BACKGROUND
# ------------------------------------------------------------

fresh_extracted = cv2.bitwise_and(
    fresh_image,
    fresh_image,
    mask=fresh_mask
)

rotten_extracted = cv2.bitwise_and(
    rotten_image,
    rotten_image,
    mask=rotten_mask
)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(fresh_extracted)
plt.title("Extracted Fresh Apple")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(rotten_extracted)
plt.title("Extracted Rotten Apple")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 14. FRUIT BOUNDARY EXTRACTION
# ------------------------------------------------------------

def create_boundary(image, mask):

    boundary_image = image.copy()

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        boundary_image,
        contours,
        -1,
        (255, 0, 0),
        3
    )

    return boundary_image


fresh_boundary = create_boundary(
    fresh_image,
    fresh_mask
)

rotten_boundary = create_boundary(
    rotten_image,
    rotten_mask
)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(fresh_boundary)
plt.title("Fresh Apple Boundary")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(rotten_boundary)
plt.title("Rotten Apple Boundary")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 15. K-MEANS COLOR SEGMENTATION
# ------------------------------------------------------------

def kmeans_segmentation(
    image,
    fruit_mask,
    k=5
):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    pixels = lab[
        fruit_mask > 0
    ]

    pixels = np.float32(
        pixels
    )

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        pixels
    )

    label_image = np.full(
        fruit_mask.shape,
        -1,
        dtype=np.int32
    )

    label_image[
        fruit_mask > 0
    ] = labels

    return (
        label_image,
        kmeans.cluster_centers_
    )


fresh_labels, fresh_centers = (
    kmeans_segmentation(
        fresh_image,
        fresh_mask,
        k=5
    )
)

rotten_labels, rotten_centers = (
    kmeans_segmentation(
        rotten_image,
        rotten_mask,
        k=5
    )
)


# ------------------------------------------------------------
# 16. DISPLAY K-MEANS CLUSTERS
# ------------------------------------------------------------

def display_clusters(
    image,
    labels,
    k=5
):

    plt.figure(figsize=(15, 7))

    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    for i in range(k):

        cluster_mask = (
            labels == i
        )

        plt.subplot(
            2,
            3,
            i + 2
        )

        plt.imshow(
            cluster_mask,
            cmap="gray"
        )

        plt.title(
            f"Cluster {i}"
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()


display_clusters(
    rotten_image,
    rotten_labels,
    k=5
)


# ------------------------------------------------------------
# 17. SELECT DEFECT REGION
# ------------------------------------------------------------

# Cluster 0 is selected as the
# visible defect/discoloration region.

rotten_defect_mask = (
    rotten_labels == 0
).astype(np.uint8) * 255


plt.figure(figsize=(6, 5))

plt.imshow(
    rotten_defect_mask,
    cmap="gray"
)

plt.title(
    "Rotten Apple 20 - Cluster 0 Defect Mask"
)

plt.axis("off")
plt.show()


# ------------------------------------------------------------
# 18. MORPHOLOGICAL CLEANING
# ------------------------------------------------------------

open_kernel = np.ones(
    (5, 5),
    np.uint8
)

close_kernel = np.ones(
    (11, 11),
    np.uint8
)

rotten_defect_clean = cv2.morphologyEx(
    rotten_defect_mask,
    cv2.MORPH_OPEN,
    open_kernel
)

rotten_defect_clean = cv2.morphologyEx(
    rotten_defect_clean,
    cv2.MORPH_CLOSE,
    close_kernel
)


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(
    rotten_defect_mask,
    cmap="gray"
)
plt.title("Original Defect Mask")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(
    rotten_defect_clean,
    cmap="gray"
)
plt.title("Cleaned Defect Mask")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 19. HIGHLIGHT DEFECT REGIONS
# ------------------------------------------------------------

defect_highlight = rotten_image.copy()

contours, _ = cv2.findContours(
    rotten_defect_clean,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(
    defect_highlight,
    contours,
    -1,
    (255, 0, 0),
    3
)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(rotten_image)
plt.title("Original Rotten Apple")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(defect_highlight)
plt.title("Visible Defect Highlighted")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 20. FINAL DEFECT VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(rotten_image)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(
    rotten_defect_clean,
    cmap="gray"
)
plt.title("Cluster 0 - Defect Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(defect_highlight)
plt.title("Highlighted Defect")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 21. CALCULATE VISIBLE DEFECT PERCENTAGE
# ------------------------------------------------------------

fruit_area = np.count_nonzero(
    rotten_mask
)

defect_area = np.count_nonzero(
    rotten_defect_clean
)

if fruit_area > 0:

    defect_percentage = (
        defect_area /
        fruit_area
    ) * 100

else:

    defect_percentage = 0


print("Fruit Area:", fruit_area)
print("Visible Defect Area:", defect_area)
print(
    f"Visible Defect Percentage: "
    f"{defect_percentage:.2f}%"
)


# ------------------------------------------------------------
# 22. FINAL RESULT
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(rotten_image)
plt.title("Original Rotten Apple 20")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(defect_highlight)

plt.title(
    f"Visible Defect: "
    f"{defect_percentage:.2f}%"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 23. DEFECT SEVERITY
# ------------------------------------------------------------

if defect_percentage <= 5:

    severity = "Very Low"

elif defect_percentage <= 15:

    severity = "Low"

elif defect_percentage <= 30:

    severity = "Moderate"

else:

    severity = "High"


print(
    "Project-defined Visible Damage Level:",
    severity
)


# ------------------------------------------------------------
# 24. DEFECT DISTRIBUTION
# ------------------------------------------------------------

height, width = rotten_mask.shape

top_region = rotten_defect_clean[
    :height // 3,
    :
]

middle_region = rotten_defect_clean[
    height // 3 : 2 * height // 3,
    :
]

bottom_region = rotten_defect_clean[
    2 * height // 3 :,
    :
]

regions = {

    "Top": np.count_nonzero(
        top_region
    ),

    "Middle": np.count_nonzero(
        middle_region
    ),

    "Bottom": np.count_nonzero(
        bottom_region
    )
}


most_damaged_region = max(
    regions,
    key=regions.get
)


print("\nVisible Defect Distribution")

for region, value in regions.items():

    print(
        f"{region}: {value} pixels"
    )

print(
    "\nMost affected region:",
    most_damaged_region
)


# ------------------------------------------------------------
# 25. DEFECT DISTRIBUTION GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    regions.keys(),
    regions.values()
)

plt.title(
    "Visible Defect Distribution"
)

plt.xlabel(
    "Fruit Region"
)

plt.ylabel(
    "Defect Pixels"
)

plt.show()


# ------------------------------------------------------------
# 26. AUTOMATIC QUALITY REPORT
# ------------------------------------------------------------

print("\n")
print("=" * 50)
print("        SMARTFRUIT VISION REPORT")
print("=" * 50)

print("\nImage:")
print("Rotten Apple 20")

print(
    f"\nVisible Defect Percentage:"
    f" {defect_percentage:.2f}%"
)

print(
    f"Visible Damage Level:"
    f" {severity}"
)

print(
    f"Most Affected Region:"
    f" {most_damaged_region}"
)

if severity == "Very Low":

    assessment = (
        "Very small visible surface damage detected."
    )

elif severity == "Low":

    assessment = (
        "Low visible surface damage detected."
    )

elif severity == "Moderate":

    assessment = (
        "Moderate visible surface damage detected."
    )

else:

    assessment = (
        "High visible surface damage detected."
    )


print(
    f"\nAssessment:\n{assessment}"
)

print(
    "\nNote: This system analyzes visible "
    "surface regions only."
)

print(
    "It is not a food-safety or spoilage diagnosis."
)

print("=" * 50)
