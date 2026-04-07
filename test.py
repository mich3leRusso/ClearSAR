from RFI_dataset import RFIDataset

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import json
from pathlib import Path
from PIL import Image
from utils.show_bbx import show_image_with_boxes
from transformation import test_transform


def test_model(model, test_dir: str, out_dir: str, batch_size: int = 1,
                save_images: bool = True,
               score_threshold: float = 0.5
               ):

    test_list = list(sorted(test_dir.glob("*.png")))

    test_data   = RFIDataset(test_list, None, transforms=test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda x: tuple(zip(*x)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    detections = []

    # Create output directory for images
    if save_images:
        img_out_dir = Path(out_dir).parent / "detections_vis"
        img_out_dir.mkdir(parents=True, exist_ok=True)

    for imgs, _, img_paths, orig_shapes in tqdm(test_loader, total=len(test_loader)):
        imgs = [img.to(device) for img in imgs]

        with torch.no_grad():
            outputs = model(imgs)

        for output, img_path, orig_shape in zip(outputs, img_paths, orig_shapes):
            boxes  = output['boxes'].cpu()
            scores = output['scores'].cpu()

            orig_h, orig_w = orig_shape[0], orig_shape[1]
            scale_x = orig_w / 336
            scale_y = orig_h / 512

            img_detections = []
            for box, score in zip(boxes, scores):
                x_min, y_min, x_max, y_max = box

                x_min = float(x_min) * scale_x
                x_max = float(x_max) * scale_x
                y_min = float(y_min) * scale_y
                y_max = float(y_max) * scale_y

                det = {
                    "image_id":    int(img_path.stem),
                    "category_id": 1,
                    "bbox":  [x_min, y_min, x_max - x_min, y_max - y_min],  # COCO: [x,y,w,h]
                    "score": float(score)
                }
                detections.append(det)
                img_detections.append(det)

            if save_images:
                filtered = [(det["bbox"], det["score"]) for det in img_detections
                            if det["score"] >= score_threshold]
                filtered_boxes  = [f[0] for f in filtered]
                filtered_scores = [f[1] for f in filtered]

                pil_img  = Image.open(img_path)
                annotated = show_image_with_boxes(
                    pil_img, filtered_boxes,
                    scores=filtered_scores,
                    show_label=True,       # ← toggle label on/off here
                    return_image=True
                )
                annotated.save(img_out_dir / img_path.name)

    with open(out_dir, "w") as f:
        json.dump(detections, f)
