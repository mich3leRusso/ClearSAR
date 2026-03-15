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
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from test import test_model
from train import train_one_epoch, evaluate

def main(verbose: bool = False, validation:float= 0.0, train: bool= True, data_dir:str="ClearSAR/data"):

    data_dir = Path(data_dir)
    train_dir = data_dir / "images/train"
    test_dir = data_dir / "images/test"
    out_dir= data_dir / "test_evaluation"
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

    # Split into train/val (80/20), keeping pairs aligned
    train_imgs, val_imgs, train_targets, val_targets = train_test_split(
        train_list,
        targets_list,
        test_size=0.2,
        random_state=42
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = RFIDataset(train_imgs, train_targets, transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)), 
        
    )

    if validation>0.0:
        val_dataset=RFIDataset(val_imgs, val_targets,transform)
        val_dataloader= DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False, 
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
    if train:
    
        best_loss = float("inf")
        
        for epoch in range(1, num_epochs + 1):
            avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
            scheduler.step()
            avg_val_loss   = evaluate(model, val_dataloader, device)
            print(f"\n── Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}\n")
            input()
            if avg_val_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), model_dir / "satmae_rcnn_best.pth")

        # Save final checkpoint
        torch.save(model.state_dict(), model_dir / "satmae_rcnn_final.pth")
    else :

        # Load the best saved checkpoint
        checkpoint_path = model_dir / "satmae_rcnn_best.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    #evaluate the model 
    test_model(model, test_dir=test_dir, out_dir=out_dir)



if __name__ == "__main__":
    parser=ArgumentParser()
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--validation",  # fixed typo: valdation → validation
        type=float,
        default=0.2,
        help="Fraction of data to use as validation set"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model; if not set, loads a pretrained model"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset directory"
    )
    args = parser.parse_args()

    main(verbose=args.verbose, validation=args.validation, train=args.train)