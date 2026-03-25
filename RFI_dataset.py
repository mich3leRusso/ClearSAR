import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Callable, Any
import numpy as np

class RFIDataset(Dataset):
    """
    PyTorch Dataset for loading RFI (Radio Frequency Interference) detection images and their annotations.

    This dataset class handles loading images and their corresponding bounding box annotations,
    with optional image transformations.
    """

    def __init__(
            self,
            images: list[Path],
            targets: list[dict[str, torch.Tensor]] | None = None,
            transforms: Callable[[Any], Any] | None = None,
            verbose: bool = False
        ):
        self.images = images
        self.targets = targets
        self.transforms = transforms
        self.verbose = verbose

    def __getitem__(self, idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        img = np.array(Image.open(self.images[idx]).convert("RGB"))
        orig_shape = img.shape  # (H, W, C)

        target = self.targets[idx] if self.targets is not None else None

        if self.transforms:
            if target is not None and target["boxes"].numel() > 0:
                # Train: pass boxes and labels to Albumentations
                boxes  = target["boxes"].numpy().tolist()
                labels = target["labels"].numpy().tolist()
                transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
                boxes  = transformed["bboxes"]
                labels = transformed["labels"]
                target = {
                    "boxes":  torch.tensor(boxes,  dtype=torch.float32) if len(boxes) > 0
                            else torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.int64) if len(labels) > 0
                            else torch.zeros(0, dtype=torch.int64)
                }
            else:
                # Test: no boxes, just transform the image
                transformed = self.transforms(image=img)

            img = transformed["image"]

        return img.to(device), target, self.images[idx], orig_shape
    
    def __len__(self) -> int:
        return len(self.images)

    def n_images_w_boxes(self):
        if self.targets is None:
            raise ValueError("Targets are not available for counting.")
        return sum(1 for target in self.targets if target["boxes"].numel() > 0)
