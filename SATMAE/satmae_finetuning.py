import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from model import  MaskedAutoencoderViT
from torchview import draw_graph

class SatMAE_Encoder(nn.Module):

    def __init__(self, embed_dim=1024, out_channels=256):
        super().__init__()
        
        self.model = MaskedAutoencoderViT.from_pretrained("MVRL/satmaepp_ViT-L_pretrain_fmow_rgb")
        
        #SATMAE encoder 
        #self.patch_embed = model.patch_embed
        
        #self.cls_token = model.cls_token
        #self.pos_embed = self.pos_embed

        #self.blocks = model.blocks
        #self.norm = model.norm

        
        # Simple Feature Pyramid Network (FPN) neck
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
        # Scale 4: 1/64 resolution (downsample from 7x7 to 4x4)
        self.scale4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # 1. Forward pass through SatMAE encoder (ensure 0% masking for detection)
        # SatMAE forward_encoder returns (latent, mask, ids_restore)
        latent_features = self.model.forward_encoder(x, mask_ratio=0.0) 
        
        # 2. Slice off the [CLS] token at index 0
        # Shape goes from [B, 197, 1024] to [B, 196, 1024]
        patch_tokens = latent_features[:, 1:, :] 
        
        # 3. Reshape 1D sequence to 2D spatial map
        B = patch_tokens.shape[0]
        # Calculate grid dynamically (sqrt of 196 = 14)
        grid_size = int(patch_tokens.shape[1] ** 0.5) 
        
        # Transpose and reshape to [B, 1024, 14, 14]
        spatial_map = patch_tokens.permute(0, 2, 1).reshape(B, 1024, grid_size, grid_size)
        
        # 4. Generate the FPN Multi-Scale Outputs
        out = OrderedDict()
        out['0'] = self.scale1(spatial_map)  # Stride 8:  [B, 256, 28, 28]
        out['1'] = self.scale2(spatial_map)  # Stride 16: [B, 256, 14, 14]
        out['2'] = self.scale3(spatial_map)  # Stride 32: [B, 256, 7, 7]
        out['3'] = self.scale4(out['2'])     # Stride 64: [B, 256, 4, 4]
        
        return out

class SatMAE_RCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        #Encoder 
        self.encoder= SatMAE_Encoder()

        for param in self.encoder.model.parameters():
            param.requires_grad = False
        
        #Define matching anchor generators for the 4 feature maps generated above
        self.anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,)),
            aspect_ratios=tuple([(0.5, 1.0, 2.0) for _ in range(4)])
        )

        self.encoder.out_channels = 256

        # Initialize the final bounding box detector (adjust num_classes as needed)
        self.detector = FasterRCNN(
            self.encoder,
            num_classes=num_classes+1, # e.g., 1 class + 1 background
            rpn_anchor_generator=self.anchor_generator, 
            min_size=224, 
            max_size=224
        )
    def forward(self, x):
        return self.detector(x)



if __name__=="__main__":
    #model=SatMAE_Encoder()
    #print(model(torch.randn(2, 3, 224, 224)))

    model=SatMAE_RCNN()
    model.eval()
    print(model(torch.randn(2, 3, 224, 224)))
