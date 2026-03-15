from RFI_dataset import RFIDataset

from torchvision import transforms
from torch.utils.data import DataLoader
import torch  
from tqdm import tqdm
import json
from PIL import Image
from utils.show_bbx import show_image_with_boxes

def test_model(model, test_dir: str, out_dir: str, batch_size: int = 1, verbose: bool = False):

    test_list = list(sorted(test_dir.glob("*.png"))) 
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_data = RFIDataset(test_list, None, transforms=transformation)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    detections = []
    idx = 0  

    for imgs, _ in tqdm(test_loader, total=len(test_loader)):
        imgs = [img.to(device) for img in imgs]

        img_paths = test_list[idx: idx + len(imgs)]
        idx += len(imgs)

        with torch.no_grad():
            outputs = model(imgs)

        for output, img_path in zip(outputs, img_paths):
            boxes  = output['boxes'].cpu()
            scores = output['scores'].cpu()
            for box, score in zip(boxes, scores):
                x_min, y_min, x_max, y_max = box
                detections.append({
                    "image_id": int(img_path.stem),
                    "category_id": 1,
                    "bbox": list(map(float, [x_min, y_min, x_max - x_min, y_max - y_min])),
                    "score": float(score)
                })

    with open(out_dir, "w") as f:
        json.dump(detections, f)

    if verbose:
        threshold = 0.3
        for sample_img_id in [10, 90, 356]:
            sample_img_path = test_dir / f"{sample_img_id}.png"
            sample_img_bboxes = [det["bbox"] for det in detections if det["image_id"] == sample_img_id and det["score"] >= threshold]
            show_image_with_boxes(Image.open(sample_img_path), sample_img_bboxes)
