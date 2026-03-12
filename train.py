import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss, box_iou
from transformers import  AutoImageProcessor
from torch.optim import AdamW, lr_scheduler
from Dino_Detection import EomtDetector
from torchview import draw_graph
import graphviz
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

def train(
    train_loader,
    lr: float = 1e-4, 
    val_loader= None ,
    num_classes:   int   = 1,
    num_epochs:    int   = 1,
    device:        str   = "cuda",
    freeze_backbone: bool = True,
    validation: bool =True, 
    model_id: str = "tue-mps/coco_instance_eomt_large_640",
    save_path: str = "eomt_finetuned.pt",
):
    processor = AutoImageProcessor.from_pretrained(model_id)


    model = EomtDetector(model_id, num_classes, freeze_backbone)

    model_graph = draw_graph(model, input_size=(1, 3, 640, 640), device="cuda")
    model_graph.visual_graph.render("model_graph", format="png", cleanup=True)
    input()
    # Separate LRs: higher for heads, lower (or zero) for any unfrozen backbone
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("model.model.")]
    optimizer = AdamW(head_params, lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    if validation:
        best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            mask_labels  = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]

            outputs = model(pixel_values, mask_labels, class_labels)
            
            loss = outputs.loss  # Hungarian-matched: CE + mask BCE + dice

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # ── Validate ──
        if validation:

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device)
                    mask_labels  = [m.to(device) for m in batch["mask_labels"]]
                    class_labels = [c.to(device) for c in batch["class_labels"]]
                    outputs = model(pixel_values, mask_labels, class_labels)
                    val_loss += outputs.loss.item()

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Saved best model → {save_path}")


    return model