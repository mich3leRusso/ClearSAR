# ClearSAR RFI Detection Challenge

---
name: ClearSAR
license: Non-comercial use
source: https://platform.ai4eo.eu/
thumbnail: https://images.unsplash.com/photo-1534294228306-bd54eb9a7ba8?q=80&w=2080&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D
authors:
  - Jakub Nalepa, KP Labs, Poland
  - Krzysztof Kotowski, KP Labs, Poland
  - Bartosz Grabowski, KP Labs, Poland
  - Panče Panov, Bias Variance Labs, Slovenia
  - Tadej Tomanič, Bias Variance Labs, Slovenia
  - Alice Baudhuin, Bias Variance Labs, Slovenia
  - Jan Sotošek, Bias Variance Labs, Slovenia
  - Kevin Halsall, Telespazio UK, UK
  - James Harding, Telespazio UK, UK
  - Leonardo De Laurentiis, ESA, Italy
  - Roberto Del Prete, ESA, Italy
  - Lorenzo Papa, ESA, Italy
  - Gabriele Meoni, ESA, Italy
---
The objective of the **ClearSAR Challange** is to build a method, based on the provided training set of quicklooks (RGB images in a png format) accompanied by ground-truth annotations (bounding boxes), to **detect & localize RFI in quicklook images**. RFI objects are of different characteristics – these difficulties are reflected in the released dataset.

The participants are provided with the AI-ready dataset (RFI dataset), comprising **3,940 quicklook (RGB) images**. The dataset was divided into training and test sets based on multiple criteria to ensure balanced representation across geographic regions, RFI sizes, and types of interference.

The training set contains **3,154 images**, accompanied with the ground-truth annotations (bounding boxes showing RFI artifacts), whereas the test set contains **786 images**. Furthermore, the validation set includes 50% of test images.

The score obtained by a Team is calculated using **mean Average Precision (mAP)**. It is calculated at 10 Intersection-over-Union (IoU) thresholds of .50:.05:.95, and mAP ranges from 0 to 1, with 1 indicating the perfect score. The script used to calculate mAP is released as part of the Starter Pack.

## Getting Started

### Installation

Install the required dependencies with:

```
pip install -r requirements.txt
```

Download the dataset:

```
eotdl datasets get ClearSAR -v 1 --assets -p <path-installation>
```

## YOLO:

### Dataset setup
1. rename the old dataset such that the path is: 
```./ClearSAR/coco_dataset/annotations```
```./ClearSAR/coco_dataset/images```
2. run:
```
python coco_to_yolo -i ./ClearSAR/coco_dataset -o ./ClearSAR/yolo_dataset --val_split 0.2
```
### Run yolo

1. change dataset paths in: 
```configs/yolo_train.yaml```
```/home/simone/myprojects/ClearSAR/ClearSAR/yolo_dataset/data.yaml```
2. run:
```
python yolo.py --mode "train" --config configs/yolo_train.yaml
```
