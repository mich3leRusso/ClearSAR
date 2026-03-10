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
        return len(self.images)

    def n_images_w_boxes(self):
        if self.targets is None:
            raise ValueError("Targets are not available for counting.")
        return sum(1 for target in self.targets if target["boxes"].numel() > 0)

    @staticmethod
    def boxes_to_masks(
        boxes: torch.Tensor,
        labels: torch.Tensor,
        img_h: int = 342,
        img_w: int = 516,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert bounding boxes to binary instance masks for EoMT training.

        Args:
            boxes:  (N, 4) float tensor in [x1, y1, x2, y2] format (already scaled).
            labels: (N,)   long tensor of class indices.
            img_h:  target mask height (should match resized image height).
            img_w:  target mask width  (should match resized image width).

        Returns:
            mask_labels:  (N, img_h, img_w) float tensor — binary mask per instance.
            class_labels: (N,) long tensor — class index per instance (unchanged).
        """
        N = boxes.shape[0]
        mask_labels = torch.zeros((N, img_h, img_w), dtype=torch.float32)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Clamp and round to valid pixel indices
            x1 = max(0, int(x1.item()))
            y1 = max(0, int(y1.item()))
            x2 = min(img_w, int(x2.item() ))
            y2 = min(img_h, int(y2.item() ))
            mask_labels[i, y1:y2, x1:x2] = 1.0

        return mask_labels, labels.long()
