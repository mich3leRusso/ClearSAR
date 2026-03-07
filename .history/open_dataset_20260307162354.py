import json
import itertools
import random
from pathlib import Path
from typing import Callable, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# Define dataset and model paths.
data_dir = Path("data")
train_dir = data_dir / "images/train"
test_dir = data_dir / "images/test"
model_dir = Path("models")
out_dir = Path("submission.json")
train_labels_path = data_dir / "annotations/instances_train.json"

# Training configuration: simple defaults used for the example run.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 3
num_iter = 20
batch_size = 8

# Random seed for reproducibility of experiments
seed = 0