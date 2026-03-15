from RFI_dataset import RFIDataset

from torchvision import transforms
from torch.utils.data import DataLoader
import torch  
import tqdm
import json
from PIL import Image
from utils.show_bbx import show_image_with_boxes

def test_model(model, test_dir:str,  out_dir:str, batch_size : int =1, verbose:bool =False):
    
    test_list = list(test_dir.iterdir())
    transformation= transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_data = RFIDataset(test_list, None, transforms=transformation)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    detections = []
    for i, (imgs, img_paths) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
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

    with open(out_dir, "w") as f:
        json.dump(detections, f)

    if verbose:
        sample_img_ids = [10, 90, 356]
        threshold = 0.3

        for sample_img_id in sample_img_ids:
            sample_img_path = test_dir / f"{sample_img_id}.png"
            sample_img_bboxes = [det["bbox"] for det in detections if det["image_id"] == sample_img_id and det["score"] >= threshold]
            sample_img = Image.open(sample_img_path)
            show_image_with_boxes(sample_img, sample_img_bboxes)
