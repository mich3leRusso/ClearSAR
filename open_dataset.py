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

#define model and move it to the device (GPU if available)
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
model.to(device)
model.train()

#train the model

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}")
    # Only iterate a limited number of batches (num_iter) to keep example runs fast
    for imgs, targets, _ in tqdm(itertools.islice(train_loader, num_iter), total=num_iter):
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass (model returns loss dict when in training mode)
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation step
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done, loss={losses.item():.4f}")
model_dir.mkdir(exist_ok=True)
# Save model weights for later use.
torch.save(model.state_dict(), model_dir / "model.pth")





## Inference on test set and prepare submission in COCO format.
test_list = list(test_dir.iterdir())
test_data = RFI_dataset.RFIDataset(test_list, None, transforms=T.ToTensor())
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

model.eval()
detections = []
for i, (imgs, _, img_paths) in tqdm(enumerate(test_loader), total=len(test_loader)):
    imgs = list(img.to(device) for img in imgs)
    with torch.no_grad():
        outputs = model(imgs)
    # `outputs` is a list of dicts (one per input image) containing 'boxes', 'labels' and 'scores'
    for output, img_path in zip(outputs, img_paths):
        boxes = output['boxes'].cpu()
        scores = output['scores'].cpu()
        for box, score in zip(boxes, scores):
            x_min, y_min, x_max, y_max = box
            # Convert to COCO [x, y, w, h] format
            box = [x_min, y_min, x_max - x_min, y_max - y_min]
            detections.append({
                "image_id": int(img_path.stem),
                "category_id": 1,
                "bbox": list(map(float, box)),
                "score": float(score)
            })

## save detections to a JSON file in COCO format for submission
with open(out_dir, "w") as f:
    json.dump(detections, f)

## visualize some sample detections on the test set (using a low confidence threshold to show more boxes)

sample_img_ids = [10, 90, 356]
threshold = 0.3

for sample_img_id in sample_img_ids:
    sample_img_path = test_dir / f"{sample_img_id}.png"
    sample_img_bboxes = [det["bbox"] for det in detections if det["image_id"] == sample_img_id and det["score"] >= threshold]
    sample_img = Image.open(sample_img_path)
    show_image_with_boxes(sample_img, sample_img_bboxes)


# evaluation on a few training images to check mAP metric (using COCOeval)
model.eval()
detections_sample = []
img_ids_sample = []
for i, (imgs, _, img_paths) in tqdm(enumerate(itertools.islice(train_loader, 3)), total=3):
    imgs = list(img.to(device) for img in imgs)
    with torch.no_grad():
        outputs = model(imgs)
    for output, img_path in zip(outputs, img_paths):
        img_ids_sample.append(int(img_path.stem))
        boxes = output['boxes'].cpu()
        scores = output['scores'].cpu()
        for box, score in zip(boxes, scores):
            x_min, y_min, x_max, y_max = box
            box = [x_min, y_min, x_max - x_min, y_max - y_min]
            detections_sample.append({
                "image_id": int(img_path.stem),
                "category_id": 1,
                "bbox": list(map(float, box)),
                "score": float(score)
            })

coco_gt = COCO(train_labels_path)
coco_dt = coco_gt.loadRes(detections_sample)
evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
evaluator.params.imgIds = img_ids_sample
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()
print("mAP metric:", float(evaluator.stats[0]))
