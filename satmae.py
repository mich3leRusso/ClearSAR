from huggingface_hub import hf_hub_download
import torch
from torchview import draw_graph
from SATMAE.model import MaskedAutoencoderViT


model = MaskedAutoencoderViT.from_pretrained("MVRL/satmaepp_ViT-L_pretrain_fmow_rgb")
#FORWARD USES ONLY THE FORWARD PART

print(model.forward_encoder(torch.randn(1, 3, 224, 224), mask_ratio=0.0).shape)