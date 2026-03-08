import os
import numpy as np
from PIL import Image

def compute_mean_std(folder_path):
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    total_pixels = 0

    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

    print(f"Found {len(image_files)} images...")

    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        img = np.array(Image.open(img_path).convert('RGB')).astype(np.float64)

        h, w, _ = img.shape
        total_pixels += h * w

        pixel_sum += img.sum(axis=(0, 1))
        pixel_sq_sum += (img ** 2).sum(axis=(0, 1))

    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_sq_sum / total_pixels - mean ** 2)

    # Normalize to [0, 1] range
    mean /= 255.0
    std /= 255.0

    return mean, std


if __name__ == "__main__":

    folder = "/home/simone/myprojects/ClearSAR/ClearSAR/yolo_dataset/images/train"
    mean, std = compute_mean_std(folder)

    print(f"\nMean (RGB): {mean}")
    print(f"Std  (RGB): {std}")

    # Mean (RGB): [0.25503881 0.34011768 0.26059022]
    # Std  (RGB): [0.15046017 0.21711302 0.10183406]