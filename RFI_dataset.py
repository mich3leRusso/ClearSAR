import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Callable, Any

class RFIDataset(Dataset):
    """
    PyTorch Dataset for loading RFI (Radio Frequency Interference) detection images and their annotations.

    This dataset class handles loading images and their corresponding bounding box annotations,
    with optional image transformations.
    """

    def __init__(
            self,
            images: list[Path],
            targets: list[dict[str, torch.Tensor]] | None=None,
            transforms: Callable[[Any], Any] | None=None
        ):
        """
        Initialize the RFIDataset.

        Args:
            images (list): List of image file paths to load.
            targets (list, optional): List of target dictionaries containing 'boxes' and 'labels' tensors.
                                     If None, the dataset operates in inference mode. Defaults to None.
            transforms (callable, optional): Optional image transformation function to apply. Defaults to None.
        """
        self.images = images
        self.targets = targets
        self.transforms = transforms

    def __getitem__(self, idx: int) -> tuple[Image.Image | torch.Tensor, dict | None, Path]:
        """
        Get an image, its target annotations, and file path by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - img (PIL.Image or Tensor): The image, optionally transformed.
                - target (dict or None): Dictionary with 'boxes' and 'labels' tensors, or None if targets not provided.
                - img_path (Path): Path to the image file.
        """
        img = Image.open(self.images[idx])
        target = self.targets[idx] if self.targets is not None else None
        if self.transforms:
            img = self.transforms(img)
        print(img.shape)
        return img, target, self.images[idx]

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.images)

    def n_images_w_boxes(self):
        """
        Count the number of images with bounding boxes.
        """
        if self.targets is None:
            raise ValueError("Targets are not available for counting.")
        return sum(1 for target in self.targets if target["boxes"].numel() > 0)
