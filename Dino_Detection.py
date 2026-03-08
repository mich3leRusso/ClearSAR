import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import FeaturePyramidNetwork
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests


class DINOv3BackboneWrapper(nn.Module):
    """Wraps DINOv3 to extract multi-scale patch features for FPN."""

    def __init__(self, backbone, out_channels: int = 256):
        super().__init__()
        self.backbone = backbone
        hidden_dim = backbone.config.hidden_size  # e.g. 1024 for ViT-L

        # Project each selected layer's tokens to out_channels for FPN
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
        # hidden_states: tuple of (B, 1+num_patches, hidden_dim) per layer
        hidden_states = outputs.hidden_states  # all transformer layers

        B = x.shape[0]
        H = W = int((hidden_states[0].shape[1] - 1) ** 0.5)  # exclude CLS

        feature_map = {}
        for key, layer_idx in [("layer_3", 3), ("layer_6", 6), ("layer_9", 9), ("layer_12", 12)]:
            tokens = hidden_states[layer_idx][:, 1:, :]        # (B, N, C) — drop CLS
            spatial = tokens.permute(0, 2, 1).reshape(B, -1, H, W)  # (B, C, H, W)
            feature_map[key] = self.projections[key](spatial)

        return feature_map  # OrderedDict of 4 feature maps at different depths


class DinoRCNN(nn.Module):

    def __init__(self, num_classes: int = 1, freeze_backbone: bool = True):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

        dino_backbone = AutoModel.from_pretrained(
            pretrained_model_name,
            device_map=device,  
        )

        if freeze_backbone:
            for param in dino_backbone.parameters():
                param.requires_grad = False

        # Wrap backbone to output multi-scale feature maps
        self.backbone_wrapper = DINOv3BackboneWrapper(dino_backbone, out_channels=256)
        self.backbone_wrapper.out_channels = 256  # required by FasterRCNN

        # FPN on top of multi-scale DINOv3 features
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 256, 256, 256],
            out_channels=256,
        )

        # Anchor generator — one size per FPN level
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )

        # RoI pooling resolution
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["layer_3", "layer_6", "layer_9", "layer_12"],
            output_size=7,
            sampling_ratio=2,
        )

        # Full Faster R-CNN with DINOv3 backbone
        self.rcnn = FasterRCNN(
            backbone=self.backbone_wrapper,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

    def forward(self, images, targets=None):
        """
        images:  list of float tensors (C, H, W) in [0, 1]
        targets: list of dicts with 'boxes' (Nx4) and 'labels' (N,) — only during training
        """
        
        return self.rcnn(images, targets)


# --- Usage ---
if __name__ == "__main__":

    model = DinoRCNN(num_classes=1, freeze_backbone=True)
    model.eval()

    # Load and preprocess image manually to tensor for FasterRCNN
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    to_tensor = torchvision.transforms.ToTensor()
    img_tensor = to_tensor(image)  # (3, H, W) in [0, 1]

    with torch.no_grad():
        predictions = model([img_tensor])

    print("Boxes:",  predictions[0]["boxes"].shape)
    print("Labels:", predictions[0]["labels"].shape)
    print("Scores:", predictions[0]["scores"].shape)
