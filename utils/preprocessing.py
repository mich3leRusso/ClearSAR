from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

import cv2
import torch
from pathlib import Path
from PIL import Image
import random

import os
from pathlib import Path
from PIL import Image
import cv2
from scipy.ndimage import uniform_filter





from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import os


def remove_vertical_stripes(
    image_path: str | Path,
    output_folder: str | Path,
    smooth_window: int = 51,
    strength: float = 1.0,
    preserve_brightness: bool = True,
) -> str:
    """
    Remove vertical stripe / banding artifacts from an RGB image.

    Method:
    1. For each channel, compute the column-wise mean profile.
    2. Smooth that profile to estimate the low-frequency illumination trend.
    3. Subtract the residual column bias (stripe component) from each column.

    Args:
        image_path: Input RGB image path.
        output_folder: Folder where output image will be saved.
        smooth_window: Odd Gaussian smoothing window across columns.
            Larger values remove only broad column bias; smaller values remove
            finer stripes. Good starting range: 31-101.
        strength: Correction strength. 1.0 = full correction,
            0.5 = gentler correction.
        preserve_brightness: Re-center output to preserve global channel means.

    Returns:
        Path to saved destriped image.
    """
    os.makedirs(output_folder, exist_ok=True)
    img = Image.open(image_path).convert('RGB')
    arr = np.asarray(img).astype(np.float32)

    if smooth_window % 2 == 0:
        smooth_window += 1

    out = np.empty_like(arr)

    for c in range(3):
        ch = arr[:, :, c]
        col_profile = ch.mean(axis=0)
        trend = cv2.GaussianBlur(
            col_profile.reshape(1, -1),
            (smooth_window, 1),
            0
        ).reshape(-1)
        stripe_component = col_profile - trend
        corrected = ch - strength * stripe_component[None, :]
        out[:, :, c] = corrected

    if preserve_brightness:
        in_mean = arr.mean(axis=(0, 1), keepdims=True)
        out_mean = out.mean(axis=(0, 1), keepdims=True)
        out += (in_mean - out_mean)

    out = np.clip(out, 0, 255).astype(np.uint8)

    stem = Path(image_path).stem
    out_path = os.path.join(output_folder, f"{stem}_destriped.png")
    Image.fromarray(out).save(out_path)
    return out_path

def convert2YCrCb(img_path: Path) -> np.ndarray:
    """
    Opens a single image and converts it from BGR to YCrCb color space.

    Args:
        img_path: Path to the image file.
    Returns:
        YCrCb image as uint8 numpy array [H, W, 3].
    """
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

def visualize_ycrcb_random(
    image_paths: List[Path],
    n_samples: int = 3,
    seed: int = None
) -> None:
    """
    Randomly samples n_samples images, converts to YCrCb,
    and displays the original + each channel side by side.

    Args:
        image_paths: List of image paths to sample from.
        n_samples:   Number of random images to display.
        seed:        Optional random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    sampled_paths = random.sample(image_paths, min(n_samples, len(image_paths)))
    converted = [convert2YCrCb(p) for p in sampled_paths]
    
    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4 * n_samples))

    # Handle single-row case
    if n_samples == 1:
        axes = [axes]

    channel_names = ["Y (Luminance)", "Cr (Red diff)", "Cb (Blue diff)"]
    channel_cmaps = ["gray", "RdBu_r", "PuBu_r"]

    for row, (img_path, img_ycrcb) in enumerate(zip(sampled_paths, converted)):

        # Col 0: Original BGR rendered as RGB
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        axes[row][0].imshow(img_rgb)
        axes[row][0].set_title(f"Original RGB\n{img_path.name}", fontsize=9)
        axes[row][0].axis("off")

        # Col 1: Full YCrCb image (displayed as-is via imshow)
        axes[row][1].imshow(img_ycrcb)
        axes[row][1].set_title("YCrCb (composite)", fontsize=9)
        axes[row][1].axis("off")

        # Cols 2-4: Individual Y, Cr, Cb channels
        for col, (ch_idx, name, cmap) in enumerate(
            zip(range(3), channel_names, channel_cmaps), start=2
        ):
            axes[row][col].imshow(img_ycrcb[:, :, ch_idx], cmap=cmap)
            axes[row][col].set_title(name, fontsize=9)
            axes[row][col].axis("off")

    plt.suptitle("YCrCb Color Space — Random Sample Visualization", fontsize=13, y=1.01)
    plt.tight_layout()
    #plt.savefig("ycrcb_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    #print("Saved → ycrcb_visualization.png")

def convert2fft(image_paths: List[Path], magnitude: bool = True) -> List[np.ndarray]:
    """
    Convert a list of images to their FFT magnitude or phase spectrum.

    Args:
        image_paths: List of paths to input images.
        magnitude:   If True, return magnitude spectrum; if False, return phase spectrum.

    Returns:
        List of normalized FFT spectra as uint8 numpy arrays (H, W).
    """
    results = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fourier = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        fourier_shift = np.fft.fftshift(fourier)

        real = fourier_shift[:, :, 0]
        imag = fourier_shift[:, :, 1]

        if magnitude:
            spectrum = 20 * np.log1p(cv2.magnitude(real, imag))
        else:
            spectrum = np.arctan2(imag, real)

        spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        results.append(spectrum)

    return results


def convert2fft_tensor(image_paths: List[Path], magnitude: bool = True) -> torch.Tensor:
    """Returns stacked FFT spectra as (N, 1, H, W) float tensor in [0, 1]."""
    spectra = convert2fft(image_paths, magnitude=magnitude)
    tensors = [torch.from_numpy(s).unsqueeze(0).float() / 255.0 for s in spectra]
    return torch.stack(tensors)


def visualize_fft(
    image_paths: List[Path],
    max_images: int = 4,
    save_path: Path | None = None,
    colormap: str = "inferno"
):
    """
    Visualize original images alongside their FFT magnitude and phase spectra.

    Args:
        image_paths: List of image paths to visualize.
        max_images:  Maximum number of images to display.
        save_path:   If provided, saves the figure to this path instead of showing.
        colormap:    Matplotlib colormap for the spectra (default: 'inferno').
    """
    paths = image_paths[:max_images]
    n = len(paths)

    fig = plt.figure(figsize=(15, 4 * n))
    fig.suptitle("FFT Analysis — Original | Magnitude Spectrum | Phase Spectrum", 
                 fontsize=14, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.4, wspace=0.3)

    magnitudes = convert2fft(paths, magnitude=True)
    phases     = convert2fft(paths, magnitude=False)

    for i, (img_path, mag, phase) in enumerate(zip(paths, magnitudes, phases)):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # --- Original image ---
        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(img_rgb)
        ax0.set_title(f"{img_path.name}\nOriginal", fontsize=9)
        ax0.axis("off")

        # --- Magnitude spectrum ---
        ax1 = fig.add_subplot(gs[i, 1])
        im1 = ax1.imshow(mag, cmap=colormap, vmin=0, vmax=255)
        ax1.set_title("Magnitude Spectrum", fontsize=9)
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # --- Phase spectrum ---
        ax2 = fig.add_subplot(gs[i, 2])
        im2 = ax2.imshow(phase, cmap="twilight", vmin=0, vmax=255)
        ax2.set_title("Phase Spectrum", fontsize=9)
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"✅ Saved visualization to {save_path}")
    else:
        plt.show()


def visualize_fft_with_annotations(
    image_paths: List[Path],
    targets: List[dict],
    max_images: int = 4,
    save_path: Path | None = None,
):
    """
    Visualize FFT magnitude with ground truth RFI bounding boxes overlaid.

    Args:
        image_paths: List of image paths.
        targets:     List of target dicts with 'boxes' (N,4) xyxy tensors.
        max_images:  Max images to show.
        save_path:   Save path or None to display.
    """
    paths   = image_paths[:max_images]
    tgts    = targets[:max_images]
    mags    = convert2fft(paths, magnitude=True)

    fig, axes = plt.subplots(max_images, 2, figsize=(12, 4 * max_images))
    if max_images == 1:
        axes = [axes]

    fig.suptitle("Original with GT Boxes | FFT Magnitude with GT Boxes",
                 fontsize=13, fontweight="bold")

    for i, (img_path, mag, target) in enumerate(zip(paths, mags, tgts)):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes   = target["boxes"].numpy() if isinstance(target["boxes"], torch.Tensor) \
                  else np.array(target["boxes"])

        for ax, display_img, cmap in zip(
            axes[i], [img_rgb, mag], [None, "inferno"]
        ):
            ax.imshow(display_img, cmap=cmap)
            for box in boxes:
                x1, y1, x2, y2 = box
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="red", facecolor="none"
                )
                ax.add_patch(rect)
            ax.axis("off")

        axes[i][0].set_title(f"{img_path.name} — Original", fontsize=9)
        axes[i][1].set_title(f"FFT Magnitude  |  RFI boxes: {len(boxes)}", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"✅ Saved to {save_path}")
    else:
        plt.show()

def show_image_with_boxes(image: Image.Image, boxes: list[list[float]]):
    """
    Display an image with bounding boxes overlaid.

    This function visualizes an image with rectangular bounding boxes drawn on top.

    Args:
        image (PIL.Image.Image): The image to display.
        boxes (list[list[float]]): List of bounding boxes in COCO format [x, y, w, h],
                                   where (x, y) is the top-left corner and (w, h) are width and height.
    """
    # Create a single subplot and show the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw each bounding box as a red rectangle
    for box in boxes:
        x, y, w, h = box
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

    # Hide axis ticks/labels for a cleaner visualization
    plt.axis('off')
    plt.show()



def apply_clahe(
    image_path: str,
    output_folder: str,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> str:
    """
    Apply CLAHE to the L channel of a LAB-converted RGB image.

    Converts the image to CIE LAB color space and applies Contrast Limited
    Adaptive Histogram Equalization (CLAHE) only on the L (luminance) channel,
    leaving hue and saturation untouched. This boosts local contrast and makes
    object boundaries more visible to YOLO without introducing color artifacts.

    Args:
        image_path:      Path to the input RGB image.
        output_folder:   Folder where the preprocessed image will be saved.
        clip_limit:      Contrast limiting threshold (default 2.0).
                         Higher values → more contrast boost but also more
                         noise amplification. Keep between 1.5 and 4.0
                         for SAR images.
        tile_grid_size:  Grid size for local histogram equalization
                         (default (8, 8)). Smaller tiles → more localized
                         enhancement, useful for small objects.

    Returns:
        str: Absolute path to the saved output image.

    Example:
        >>> out = apply_clahe("scene.png", "./preprocessed", clip_limit=2.5)
        >>> print(out)  # ./preprocessed/scene_clahe.png
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load with Pillow, ensure RGB
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img, dtype=np.uint8)

    # Convert RGB → LAB (OpenCV uses BGR internally, but input is already uint8 RGB)
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

    # Apply CLAHE only to the L channel (index 0 in LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])

    # Convert LAB → RGB
    img_out = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

    stem = Path(image_path).stem
    out_path = os.path.join(output_folder, f"{stem}_clahe.png")
    Image.fromarray(img_out).save(out_path)
    return out_path


def apply_lee_filter(
    image_path: str,
    output_folder: str,
    window_size: int = 7,
    looks: float = 1.0
) -> str:
    """
    Apply the Lee speckle filter to an RGB image (processed per channel).

    The Lee filter is the standard adaptive despeckling approach for SAR.
    It models SAR noise as multiplicative and adapts per-pixel based on the
    local signal-to-noise ratio:

        filtered = mean + W * (pixel - mean)
        W        = local_var / (local_var + noise_var)
        noise_var = mean² / ENL

    - In flat/homogeneous regions: W → 0  → output ≈ local mean (smooth)
    - In edges/targets:            W → 1  → output ≈ original  (sharp)

    This preserves object edges (important for tight bounding boxes) while
    suppressing speckle and RFI noise in the background.

    Args:
        image_path:    Path to the input RGB image.
        output_folder: Folder where the preprocessed image will be saved.
        window_size:   Local window size in pixels (default 7, must be odd).
                       Larger window → more smoothing and noise reduction,
                       but slight edge blurring. Typical range: 5–15.
        looks:         Equivalent Number of Looks (ENL). Represents the
                       effective number of independent looks in the SAR
                       image, which controls the assumed noise level:
                         - Lower ENL (1.0) → preserve more texture
                         - Higher ENL (4–9) → stronger smoothing
                       Estimate from a homogeneous region: ENL = mean²/variance.

    Returns:
        str: Absolute path to the saved output image.

    Example:
        >>> out = apply_lee_filter("scene.png", "./preprocessed",
        ...                        window_size=7, looks=2.0)
        >>> print(out)  # ./preprocessed/scene_lee_filter.png
    """
    os.makedirs(output_folder, exist_ok=True)

    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img, dtype=np.float32)

    output = np.zeros_like(img_np)

    for c in range(3):
        channel = img_np[:, :, c]

        # Local statistics via box filter
        local_mean    = uniform_filter(channel, size=window_size)
        local_sq_mean = uniform_filter(channel ** 2, size=window_size)

        # Local variance (clamp to 0 for numerical stability)
        local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)

        # Noise variance from multiplicative SAR model: sigma_n² = mean² / ENL
        noise_var = (local_mean ** 2) / max(looks, 1e-6)

        # Lee adaptive weighting
        weight = local_var / (local_var + noise_var + 1e-8)

        # Final filtered value
        output[:, :, c] = local_mean + weight * (channel - local_mean)

    output = np.clip(output, 0, 255).astype(np.uint8)

    stem = Path(image_path).stem
    out_path = os.path.join(output_folder, f"{stem}_lee_filter.png")
    Image.fromarray(output).save(out_path)
    return out_path




if __name__ == "__main__":

    with open("ClearSAR/data/annotations/instances_train.json") as f:
        gt = json.load(f)
        annot = gt["annotations"]

    train_dir = Path("ClearSAR/data/images/train")
    out_folder = Path("ClearSAR/data/images/train_preprocessed")

    train_list = []
    targets_list = []

    for img_fname in sorted(train_dir.iterdir()):
        img_id = int(img_fname.stem)
        boxes = [a["bbox"] for a in annot if a["image_id"] == img_id]

        train_list.append(img_fname)

        if boxes == []:
            targets_list.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            })
        else:
            xyxy_boxes = []
            for bbox in boxes:
                x_min, y_min, w, h = bbox
                xyxy_boxes.append([x_min, y_min, x_min + w, y_min + h])

            targets_list.append({
                "boxes": torch.tensor(xyxy_boxes, dtype=torch.float32),
                "labels": torch.tensor([1] * len(xyxy_boxes), dtype=torch.int64)
            })

    print(f"Total images:           {len(train_list)}")
    print(f"Images with RFI boxes:  {sum(1 for t in targets_list if t['boxes'].numel() > 0)}")
    print(f"Images without RFI:     {sum(1 for t in targets_list if t['boxes'].numel() == 0)}")

 #   visualize_ycrcb_random(train_list, 7)
    #best setting
    for img_path in train_list:
        p1 = remove_vertical_stripes(img_path, out_folder / "denoising2" / "destriped", smooth_window=51, strength=1.0)
        #p2 = apply_lee_filter(p1, out_folder / "denoising" / "lee", window_size=3, looks=4.0)
        p3 = apply_clahe(p1, out_folder / "denoising2" / "clahe", clip_limit=2.0, tile_grid_size=(8, 8))
        print(f"Done: {img_path.name}")