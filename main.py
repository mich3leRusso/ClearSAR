import json
from pathlib import Path
from Dino_Detection import DinoRCNN
from RFI_dataset import RFIDataset
from utils import show_image_with_boxes
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from train import train

def main(verbose: bool = False ):

    #open dataset and create dataloader
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
    torch.manual_seed(seed)

    with open(train_labels_path) as f:
        gt = json.load(f)
        annot = gt["annotations"]

    #visualize images with boxes
    if verbose:
        sample_img_ids = [10, 90, 356]

        for sample_img_id in sample_img_ids:
            sample_img_path = train_dir / f"{sample_img_id}.png"
            sample_img_bboxes = [ann["bbox"] for ann in annot if ann["image_id"] == sample_img_id]
            sample_img = Image.open(sample_img_path)
            show_image_with_boxes(sample_img, sample_img_bboxes)



    targets_list = []
    train_list = []
    
    for img_fname in train_dir.iterdir():
        img_id = int(img_fname.stem)
        boxes = [a["bbox"] for a in annot if a["image_id"] == img_id]
        train_list.append(img_fname)
        
        if boxes == []:
            # No RFIs — append empty target
            targets_list.append({
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),  # ✅ empty, correct shape
                "labels": torch.zeros((0,), dtype=torch.int64)
            })

            continue

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
    
    train_dataset = RFIDataset(train_list, targets_list, ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Number of images in train dataloader: {len(train_dataset)}")
    print(f"Number of images with boxes in train dataloader: {train_dataset.n_images_w_boxes()}")

    model = DinoRCNN(num_classes=2, freeze_backbone=True).to(device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
        pct_start=0.1,         
        anneal_strategy="cos", 
    )



    return

if __name__ == "__main__":
    main()
