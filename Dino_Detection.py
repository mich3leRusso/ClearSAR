import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from transformers import AutoImageProcessor, AutoModel


class DINOv3BackboneWrapper(nn.Module):
    """Wraps DINOv3 to extract multi-scale patch features for FPN."""

    def __init__(self, backbone, out_channels: int = 256):
        super().__init__()
        self.backbone = backbone
        hidden_dim = backbone.config.hidden_size
        self.patch_size = backbone.config.patch_size  # ✅ store patch size

        self.out_channels = out_channels
        self.projections = nn.ModuleDict({
            "layer_3":  nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            "layer_6":  nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            "layer_9":  nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            "layer_12": nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        })

    def forward(self, x: torch.Tensor):
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states

        B, C_in, H_img, W_img = x.shape

        # ✅ Compute actual patch grid from image dimensions (handles non-square images)
        H = H_img // self.patch_size
        W = W_img // self.patch_size

        feature_map = {}
        for key, layer_idx in [("layer_3", 3), ("layer_6", 6), ("layer_9", 9), ("layer_12", 12)]:
            tokens = hidden_states[layer_idx][:, 1:, :]              # drop CLS (B, H*W, C)
            spatial = tokens.permute(0, 2, 1).reshape(B, -1, H, W)  # (B, C, H, W)
            feature_map[key] = self.projections[key](spatial)

        return feature_map


class DinoRCNN(nn.Module):

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True):
        # ✅ num_classes=2: 0=background, 1=RFI — never pass 1, FasterRCNN reserves 0
        super().__init__()

        pretrained_model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

        dino_backbone = AutoModel.from_pretrained(
            pretrained_model_name,
            device_map="cpu",  # ✅ let FasterRCNN handle device placement, not HF
        )

        if freeze_backbone:
            for param in dino_backbone.parameters():
                param.requires_grad = False

        self.backbone_wrapper = DINOv3BackboneWrapper(dino_backbone, out_channels=256)
        self.backbone_wrapper.out_channels = 256

        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,)),          # ✅ 4 sizes for 4 feature maps
            aspect_ratios=((0.5, 1.0, 2.0),) * 4,
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["layer_3", "layer_6", "layer_9", "layer_12"],
            output_size=7,
            sampling_ratio=2,
        )

        self.rcnn = FasterRCNN(
            backbone=self.backbone_wrapper,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

    def forward(self, images, targets=None):
        return self.rcnn(images, targets)


# --- Usage ---
if __name__ == "__main__":
    model = DinoRCNN(num_classes=2, freeze_backbone=True)
    model.eval()

    to_tensor = torchvision.transforms.ToTensor()
    img_tensor = to_tensor(
        torchvision.io.read_image  # or just use a dummy tensor for testing
    )

    # Quick sanity check with dummy input
    dummy = [torch.rand(3, 342, 516)]  # matches your dataset image size
    with torch.no_grad():
        predictions = model(dummy)

    print("Boxes:",  predictions[0]["boxes"].shape)
    print("Labels:", predictions[0]["labels"].shape)
    print("Scores:", predictions[0]["scores"].shape)
