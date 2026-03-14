import json
from pathlib import Path
from RFI_dataset import RFIDataset
from utils.show_bbx import show_image_with_boxes
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from SATMAE.satmae_finetuning import SatMAE_RCNN
from RFI_dataset import RFIDataset


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    model.encoder.model.eval()
    total_loss = 0.0

    for i, (images, targets) in enumerate(data_loader):
        # Move images and targets to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # FasterRCNN in train mode returns a dict of losses
        loss_dict = model.detector(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        # Gradient clipping to stabilize ViT fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses.item()

        print(
            f"Epoch [{epoch}] Iter [{i+1}/{len(data_loader)}] "
            f"Loss: {losses.item():.4f} | "
            + " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
        )

    return total_loss / len(data_loader)


def main(verbose: bool = False):

    data_dir = Path("ClearSAR/data")
    train_dir = data_dir / "images/train"
    test_dir = data_dir / "images/test"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    train_labels_path = data_dir / "annotations/instances_train.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    batch_size = 2
    seed = 0
    torch.manual_seed(seed)

    with open(train_labels_path) as f:
        gt = json.load(f)
        annot = gt["annotations"]

    if verbose:
        for sample_img_id in [10, 90, 356]:
            sample_img_path = train_dir / f"{sample_img_id}.png"
            sample_img_bboxes = [ann["bbox"] for ann in annot if ann["image_id"] == sample_img_id]
            show_image_with_boxes(Image.open(sample_img_path), sample_img_bboxes)

    targets_list = []
    train_list = []

    for img_fname in train_dir.iterdir():
        img_id = int(img_fname.stem)
        boxes = [a["bbox"] for a in annot if a["image_id"] == img_id]
        train_list.append(img_fname)

        if not boxes:
            targets_list.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            })
            continue

        for i, bbox in enumerate(boxes):
            x_min, y_min, w, h = bbox
            boxes[i] = [x_min, y_min, x_min + w, y_min + h]

        targets_list.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor([1] * len(boxes), dtype=torch.int64)
        })

    assert len(train_list) == len(targets_list)

    # ✅ Fix: must be 224x224 to match SatMAE's patch_embed expectations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = RFIDataset(train_list, targets_list, transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = SatMAE_RCNN(num_classes=1).to(device)

    # ── Optimizer: only trainable (non-frozen) parameters ─────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    # ── Scheduler: cosine annealing over all epochs ────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_loss = float("inf")
    
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()

        print(f"\n── Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}\n")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_dir / "satmae_rcnn_best.pth")
            print(f"  ✅ New best model saved (loss={best_loss:.4f})")

    # Save final checkpoint
    torch.save(model.state_dict(), model_dir / "satmae_rcnn_final.pth")


if __name__ == "__main__":
    main()
