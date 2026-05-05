import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


DATASET_PATH = "seg_train/seg_train"
RESULTS_FOLDER = "results"
MAX_IMAGES_PER_CLASS = 10
IMAGE_SIZE = (150, 150)

os.makedirs(RESULTS_FOLDER, exist_ok=True)


def load_images(dataset_path):
    images = []

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)

        if not os.path.isdir(category_path):
            continue

        count = 0

        for file in os.listdir(category_path):
            if count >= MAX_IMAGES_PER_CLASS:
                break

            img_path = os.path.join(category_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, IMAGE_SIZE)
            images.append((category, img))
            count += 1

    return images


def add_noise(image):
    noisy = random_noise(image, mode="s&p", amount=0.05)
    noisy = np.array(255 * noisy, dtype=np.uint8)
    return noisy


def spatial_median_filter(image):
    return cv2.medianBlur(image, 3)


def frequency_low_pass_filter(image, radius=30):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    f = fft2(image)
    fshift = fftshift(f)

    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)

    filtered = fshift * mask
    ishift = ifftshift(filtered)
    img_back = np.abs(ifft2(ishift))

    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    return img_back


def combined_method(image):
    spatial_result = spatial_median_filter(image)
    combined_result = frequency_low_pass_filter(spatial_result)
    return combined_result


def evaluate(original, enhanced):
    psnr = peak_signal_noise_ratio(original, enhanced, data_range=255)
    ssim = structural_similarity(original, enhanced, data_range=255)
    return psnr, ssim


def save_sample_images(original, noisy, spatial, frequency, combined, index):
    plt.figure(figsize=(12, 6))

    images = [original, noisy, spatial, frequency, combined]
    titles = [
        "Original",
        "Noisy",
        "Spatial Median",
        "Frequency FFT",
        "Combined"
    ]

    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/sample_result_{index}.png")
    plt.close()


images = load_images(DATASET_PATH)

print("Total images loaded:", len(images))

results = []

for i, (category, original) in enumerate(images):
    print(f"Processing {i + 1}/{len(images)} - {category}")

    noisy = add_noise(original)

    spatial_result = spatial_median_filter(noisy)
    frequency_result = frequency_low_pass_filter(noisy)
    combined_result = combined_method(noisy)

    methods = {
        "Noisy Image": noisy,
        "Spatial Median Filter": spatial_result,
        "Frequency FFT Low-Pass": frequency_result,
        "Combined Spatial + Frequency": combined_result
    }

    for method_name, output in methods.items():
        psnr, ssim = evaluate(original, output)

        results.append({
            "Category": category,
            "Method": method_name,
            "PSNR": psnr,
            "SSIM": ssim
        })

    if i < 3:
        save_sample_images(
            original,
            noisy,
            spatial_result,
            frequency_result,
            combined_result,
            i + 1
        )


df = pd.DataFrame(results)

df.to_csv(f"{RESULTS_FOLDER}/evaluation_results.csv", index=False)

summary = df.groupby("Method")[["PSNR", "SSIM"]].mean()
summary.to_csv(f"{RESULTS_FOLDER}/summary_results.csv")

print("\nEvaluation Completed!")
print("\nAverage Results:")
print(summary)

summary.plot(kind="bar", figsize=(8, 5))
plt.title("Average PSNR and SSIM Comparison")
plt.ylabel("Score")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(f"{RESULTS_FOLDER}/evaluation_chart.png")
plt.show()