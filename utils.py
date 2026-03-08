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

if __name__ == "__main__":

    with open("ClearSAR/data/annotations/instances_train.json") as f:
        gt = json.load(f)
        annot = gt["annotations"]

    train_dir = Path("ClearSAR/data/images/train")
    train_list = []
    targets_list = []

    for img_fname in sorted(train_dir.iterdir()):
        img_id = int(img_fname.stem)
        boxes = [a["bbox"] for a in annot if a["image_id"] == img_id]

        train_list.append(img_fname)  # always append — include empty images too

        if boxes == []:
            targets_list.append({
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,),   dtype=torch.int64)
            })
        else:
            xyxy_boxes = []
            for bbox in boxes:
                x_min, y_min, w, h = bbox
                xyxy_boxes.append([x_min, y_min, x_min + w, y_min + h])

            targets_list.append({
                "boxes":  torch.tensor(xyxy_boxes, dtype=torch.float32),
                "labels": torch.tensor([1] * len(xyxy_boxes), dtype=torch.int64)
            })

    print(f"Total images:           {len(train_list)}")
    print(f"Images with RFI boxes:  {sum(1 for t in targets_list if t['boxes'].numel() > 0)}")
    print(f"Images without RFI:     {sum(1 for t in targets_list if t['boxes'].numel() == 0)}")

    # --- Visualize ---
    visualize_fft_with_annotations(
        image_paths=train_list,
        targets=targets_list,
        max_images=4,
        save_path=Path("fft_annotated.png")
    )
