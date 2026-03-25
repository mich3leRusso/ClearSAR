import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from SATMAE.model import  MaskedAutoencoderViT
from torchview import draw_graph
import torch.nn.functional as F
from torchvision.models.detection.transform import GeneralizedRCNNTransform

class SatMAE_Encoder(nn.Module):
    def __init__(self, embed_dim=1024, out_channels=256, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.model = MaskedAutoencoderViT.from_pretrained("MVRL/satmaepp_ViT-L_pretrain_fmow_rgb")
        self.model.patch_embed.strict_img_size = False
        # Scale 1: 1/8 resolution (upsample from 14x14 to 28x28)
        self.scale1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        # Scale 2: 1/16 resolution (same as original grid: 14x14)
        self.scale2 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        # Scale 3: 1/32 resolution (downsample from 14x14 to 7x7)
        self.scale3 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        #Scale 4 
        self.scale4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def _interpolate_pos_embed(self, h, w):
        """Bicubic-interpolate positional embeddings to an (h, w) patch grid."""
        pos = self.model.pos_embed          # [1, 1+N, D]
        cls_pe  = pos[:, :1, :]
        patch_pe = pos[:, 1:, :]            # [1, N, D]
        N, D = patch_pe.shape[1], patch_pe.shape[2]
        orig = int(N ** 0.5)                # e.g., 14

        if orig == h and orig == w:
            return pos                      # no-op for 224x224

        patch_pe = patch_pe.reshape(1, orig, orig, D).permute(0, 3, 1, 2)  # [1,D,orig,orig]
        patch_pe = F.interpolate(patch_pe.float(), size=(h, w),
                                 mode='bicubic', align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, h * w, D)
        return torch.cat([cls_pe, patch_pe], dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size

        # Temporarily swap pos_embed with interpolated version
        orig_pe = self.model.pos_embed
        self.model.pos_embed = nn.Parameter(
            self._interpolate_pos_embed(h, w).to(x.device),
            requires_grad=False
        )
        latent = self.model.forward_encoder(x, mask_ratio=0.0)
        self.model.pos_embed = orig_pe      # always restore

        patch_tokens = latent[:, 1:, :]    # drop CLS
        spatial_map = patch_tokens.permute(0, 2, 1).reshape(B, 1024, h, w)

        out = OrderedDict()
        out['0'] = self.scale1(spatial_map)
        out['1'] = self.scale2(spatial_map)
        out['2'] = self.scale3(spatial_map)
        out['3'] = self.scale4(out['2'])
        return out

class PassthroughTransform(GeneralizedRCNNTransform):
    """Skip resize; only normalize."""
    def resize(self, image, target):
        return image, target   # no-op

class SatMAE_RCNN(nn.Module):
    def __init__(self, num_classes=1, img_size=512):
        super().__init__()
        self.encoder = SatMAE_Encoder()
        for param in self.encoder.model.parameters():
            param.requires_grad = False

        self.anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,)),
            aspect_ratios=tuple([(0.5, 1.0, 2.0) for _ in range(4)])
        )
        self.encoder.out_channels = 256

        self.detector = FasterRCNN(
            self.encoder,
            num_classes=num_classes + 1,
            rpn_anchor_generator=self.anchor_generator,
            min_size=500,
            max_size=img_size
        )
        # Replace the transform to skip internal resizing
        self.detector.transform = PassthroughTransform(
            min_size=img_size, max_size=img_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
    def forward(self, x):
        return self.detector(x)



if __name__=="__main__":
#    model=SatMAE_Encoder()
#    print(model(torch.randn(2, 3, 544, 544)))
 #   input()
    model=SatMAE_RCNN()

    model.eval()
    print(model(torch.randn(1, 3, 512, 512)))
