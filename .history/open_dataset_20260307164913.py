import json
import itertools
import random
from pathlib import Path
from typing import Callable, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import RFI_dataset

# Define dataset and model paths.
data_dir = Path("ClearSAR/data")
train_dir = data_dir / "images/train"
test_dir = data_dir / "images/test"
model_dir = Path("models")
out_dir = Path("submission.json")
train_labels_path = data_dir / "annotations/instances_train.json"


# Training configuration: simple defaults used for the example run.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 3
num_iter = 20
batch_size = 8

# Random seed for reproducibility of experiments
seed = 0

# Set random seed to make training deterministic
# Note: full determinism in CUDA can still be difficult to guarantee.
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

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


sample_img_ids = [1260]
with open(train_labels_path) as f:
    gt = json.load(f)
annot = gt["annotations"]

for sample_img_id in sample_img_ids:
    sample_img_path = train_dir / f"{sample_img_id}.png"
    sample_img_bboxes = [ann["bbox"] for ann in annot if ann["image_id"] == sample_img_id]
    sample_img = Image.open(sample_img_path)
    show_image_with_boxes(sample_img, sample_img_bboxes)

##Data Processing - Convert COCO annotations to the format expected by the model (bounding boxes in [x_min, y_min, x_max, y_max] format and labels).

targets_list = []
train_list = []
for img_fname in train_dir.iterdir():
    img_id = int(img_fname.stem)
    boxes = [a["bbox"] for a in annot if a["image_id"] == img_id]
    if boxes == []:
        # Skip images without annotations (no RFIs)
        continue
    train_list.append(img_fname)
    for i, bbox in enumerate(boxes):
        x_min, y_min, w, h = bbox
        x_max = x_min + w
        y_max = y_min + h
        boxes[i] = [x_min, y_min, x_max, y_max]
    targets_list.append({
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor([1] * len(boxes), dtype=torch.int64)
    })

assert len(train_list) == len(targets_list)  # Sanity check: one target per image

#create dataset and dataloader

train_data = RFI_dataset.RFIDataset(train_list, targets_list, transforms=T.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))