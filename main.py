
import json
from pathlib import Path
from RFI_dataset import RFIDataset
from utils import show_image_with_boxes
import torch
from PIL import Image
from torchvision import transforms
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch):

    pixel_values, mask_labels_list, class_labels_list = [], [], []

    for img, target in batch:
        pixel_values.append(img)

        if target is not None and target["boxes"].numel() > 0:
            masks, classes = RFIDataset.boxes_to_masks(
                target["boxes"],
                target["labels"],
            )
        else:
            # No annotations: empty masks (EoMT handles no-object via queries)
            masks   = torch.zeros((0, 342, 516), dtype=torch.float32)
            classes = torch.zeros((0,),          dtype=torch.long)

        mask_labels_list.append(masks)
        class_labels_list.append(classes)

    return {
        "pixel_values": torch.stack(pixel_values),
        "mask_labels":  mask_labels_list,   # list of (N_i, H, W) — variable N
        "class_labels": class_labels_list,  # list of (N_i,)
    }


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
    batch_size = 1

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
    transform = transforms.Compose([
        transforms.Resize((640, 640)),   # (H, W)
        transforms.ToTensor(),           
    ])
    
    train_dataset = RFIDataset(train_list, targets_list, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    s=set()
    for img,target in train_loader:
        data_dict = target[0]
        # 2. Extract the 'labels' tensor and convert it to a Python list using .tolist()
        labels_list = data_dict['labels'].tolist()
        s.update(labels_list)
        

    return

if __name__ == "__main__":
    main()