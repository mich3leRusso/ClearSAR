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
            transforms: Callable[[Any], Any] | None=None, 
            verbose: bool =False
        ):
        """
        Initialize the RFIDataset.

        Args:
            images (list): List of image file paths to load.
            targets (list, optional): List of target dictionaries containing 'boxes' and 'labels' tensors.
                                     If None, the dataset operates in inference mode. Defaults to None.
            transforms (callable, optional): Optional image transformation function to apply. Defaults to None.
            verbose , if you wanna have more visualizations
        """

        self.images = images
        self.targets = targets
        self.transforms = transforms
        self.verbose=verbose

    def __getitem__(self, idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        img = Image.open(self.images[idx])
        orig_w, orig_h = img.size  # PIL gives (W, H)

        target = self.targets[idx] if self.targets is not None else None

        if self.transforms:
            img = self.transforms(img)  # now (C, new_H, new_W)

        # Scale boxes to match resized image
        if target is not None and target["boxes"].numel() > 0:
            new_h, new_w = 342, 516
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h

            boxes = target["boxes"].clone()
            boxes[:, 0] *= scale_x  # x1
            boxes[:, 2] *= scale_x  # x2
            boxes[:, 1] *= scale_y  # y1
            boxes[:, 3] *= scale_y  # y2

            target = {
                "boxes":  boxes,
                "labels": target["labels"]
            }
        
        return img.to(device), target

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
