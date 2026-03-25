from RFI_dataset import RFIDataset

from torch.utils.data import DataLoader
import torch  
from tqdm import tqdm
import json
from PIL import Image
from utils.show_bbx import show_image_with_boxes
from transformation import test_transform

def test_model(model, test_dir: str, out_dir: str, batch_size: int = 1, verbose: bool = False):

    test_list = list(sorted(test_dir.glob("*.png")))

    test_data   = RFIDataset(test_list, None, transforms=test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda x: tuple(zip(*x)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    detections = []

    for imgs, _, img_paths, orig_shapes in tqdm(test_loader, total=len(test_loader)):
        imgs = [img.to(device) for img in imgs]

        with torch.no_grad():
            outputs = model(imgs)

        for output, img_path, orig_shape in zip(outputs, img_paths, orig_shapes):
            boxes  = output['boxes'].cpu()
            scores = output['scores'].cpu()

            # orig_shape is (H, W, C) — resized input was (512, 336)
            orig_h, orig_w = orig_shape[0], orig_shape[1]
            scale_x = orig_w / 336
            scale_y = orig_h / 512

            for box, score in zip(boxes, scores):
                x_min, y_min, x_max, y_max = box

                # Rescale back to original image coordinates
                x_min = float(x_min) * scale_x
                x_max = float(x_max) * scale_x
                y_min = float(y_min) * scale_y
                y_max = float(y_max) * scale_y

                detections.append({
                    "image_id":   int(img_path.stem),
                    "category_id": 1,
                    "bbox":  [x_min, y_min, x_max - x_min, y_max - y_min],  # COCO: [x,y,w,h]
                    "score": float(score)
                })

    with open(out_dir, "w") as f:
        json.dump(detections, f)

    if verbose:
        threshold = 0.3
        for sample_img_id in [10, 90, 356]:
            sample_img_path   = test_dir / f"{sample_img_id}.png"
            sample_img_bboxes = [det["bbox"] for det in detections
                                 if det["image_id"] == sample_img_id
                                 and det["score"] >= threshold]
            show_image_with_boxes(Image.open(sample_img_path), sample_img_bboxes)