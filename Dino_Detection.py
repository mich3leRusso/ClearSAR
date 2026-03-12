import torch
import torchvision
from PIL import Image
from transformers import EomtForUniversalSegmentation, AutoImageProcessor
from torch import nn
from torchview import draw_graph
class EomtDetector(nn.Module):
    """
    Wraps EomtForUniversalSegmentation for single-class instance segmentation
    with bounding box extraction. For 1 custom class, fine-tune from the
    pretrained COCO checkpoint or train from the base config.
    """

    def __init__(
        self,
        model_id: str = "tue-mps/videomt-dinov2-small-ytvis2019",
        num_classes: int = 1,
        device: str = "cuda",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = EomtForUniversalSegmentation.from_pretrained(
            model_id,
            num_labels=10, # Replace 10 with your exact number of labels
            ignore_mismatched_sizes=True
        )
        # Optionally freeze the ViT backbone, train only the query/mask head
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "model" in name:  # ViT backbone layers
                    param.requires_grad = False

        # # Replace the classifier head for num_classes + 1 (background/no-object)
        # EomtConfig exposes hidden_size; the final classifier is a Linear layer
        #hidden_size = self.model.config.hidden_size
        #self.model.class_predictor = torch.nn.Linear(
        #    hidden_size, num_classes + 1  # +1 for "no object"
        #)

        # ✅ Reset the criterion's class weight to match new num_classes
        # no_object class gets lower weight (0.1) as in the original EoMT paper
        #empty_weight = torch.ones(num_classes + 1)
        #empty_weight[-1] = 0.1  # last index = no-object class
        #self.model.criterion.empty_weight = empty_weight
        #self.model.criterion.num_classes = num_classes
        
        self.model.to("cuda")

    
    def forward(self, pixel_values, mask_labels=None,class_labels=None):
        pixel_values = pixel_values.to(next(self.parameters()).device)
        
        if mask_labels is not None:
            mask_labels  = [m.to(next(self.parameters()).device) for m in mask_labels]
            class_labels = [c.to(next(self.parameters()).device) for c in class_labels]


        return self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels, 
            class_labels=class_labels
        )


    def predict(self, images: list, threshold: float = 0.5):
        """Run inference and return instance segmentation results."""
        target_sizes = [(img.height, img.width) for img in images]
        outputs = self.forward(images)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )
        return results  # list of dicts with 'segmentation', 'segments_info'


# --- Bounding Box Extraction Utility ---
def masks_to_boxes(segmentation_map, segments_info):
    """Convert EoMT instance segmentation output to bounding boxes."""
    boxes, labels, scores = [], [], []
    for seg in segments_info:
        mask = segmentation_map == seg["id"]
        if mask.sum() == 0:
            continue
        rows = torch.where(mask.any(dim=1))[0]
        cols = torch.where(mask.any(dim=0))[0]
        x1, y1 = cols[0].item(), rows[0].item()
        x2, y2 = cols[-1].item(), rows[-1].item()
        boxes.append([x1, y1, x2, y2])
        labels.append(seg["label_id"])
        scores.append(seg["score"])
    return boxes, labels, scores


# --- Usage ---
if __name__ == "__main__":
    detector = EomtDetector(
        model_id="tue-mps/coco_instance_eomt_large_640",
        num_classes=1,
        freeze_backbone=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Test with a PIL image (recommended interface for EoMT processor)
    from PIL import Image as PILImage
    import numpy as np

    dummy_pil = PILImage.fromarray(
        (np.random.rand(342, 516, 3) * 255).astype(np.uint8)
    )

    results = detector.predict([dummy_pil], threshold=0.5)
    seg_map = results[0]["segmentation"]        # (H, W) tensor, instance IDs
    segments = results[0]["segments_info"]      # list of dicts

    boxes, labels, scores = masks_to_boxes(seg_map, segments)
    print(f"Detected {len(boxes)} instances")
    for b, l, s in zip(boxes, labels, scores):
        print(f"  Box: {b}  Label: {l}  Score: {s:.3f}")
