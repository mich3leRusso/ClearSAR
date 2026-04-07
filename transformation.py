import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision import transforms

# ── Define transforms ──────────────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(512, 336),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
     A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                std=[0.229, 0.224, 0.225]),    # matches FasterRCNN's transform
    ToTensorV2(),
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['labels'],
    min_visibility=0.5,
))
val_transform = A.Compose([
    A.Resize(512, 336),
    A.Normalize(mean=[0.485, 0.456, 0.406],  # ✅ same as train
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['labels'],
))

test_transform = A.Compose([
    A.Resize(512, 336),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
