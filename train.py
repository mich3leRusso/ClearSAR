import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss, box_iou

def detection_loss(cls_logits, predicted_boxes, gt_boxes, gt_labels):
    """
    cls_logits:      (N, 2)  — background vs foreground
    predicted_boxes: (N, 4)  — xyxy
    gt_boxes:        (M, 4)  — xyxy
    gt_labels:       (N,)    — 0 background, 1 foreground (after matching)
    """

    # 1. Classification loss (foreground/background)
    cls_loss = F.cross_entropy(cls_logits, gt_labels)

    # 2. Box regression only on foreground predictions
    fg_mask = gt_labels == 1
    if fg_mask.sum() > 0:
        # Smooth L1 — coordinate regression
        smooth_l1 = F.smooth_l1_loss(
            predicted_boxes[fg_mask],
            gt_boxes[fg_mask],
            beta=1.0
        )
        # GIoU — direct overlap correctness
        giou = generalized_box_iou_loss(
            predicted_boxes[fg_mask],
            gt_boxes[fg_mask],
            reduction="mean"
        )
    else:
        smooth_l1 = torch.tensor(0.0)
        giou = torch.tensor(0.0)

    total_loss = cls_loss + smooth_l1 + 2.0 * giou  # upweight GIoU
    return total_loss, {"cls": cls_loss, "smooth_l1": smooth_l1, "giou": giou}

def train(model , dataloader, optimizer, device):

    model.train()
    total_loss = 0.0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        gt_boxes = torch.stack([t["boxes"].to(device) for t in targets])
        gt_labels = torch.stack([t["labels"].to(device) for t in targets])

        cls_logits, predicted_boxes = model(images)
        loss, loss_dict = detection_loss(cls_logits, predicted_boxes, gt_boxes, gt_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Average training loss: {avg_loss:.4f}")
