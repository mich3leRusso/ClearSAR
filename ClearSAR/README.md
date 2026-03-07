---
name: ClearSAR
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
license: Non-comercial use
source: https://platform.ai4eo.eu/
---
The objective of the **ClearSAR Challange** is to build a method, based on the provided training set of quicklooks (RGB images in a png format) accompanied by ground-truth annotations (bounding boxes), to **detect & localize RFI in quicklook images**. RFI objects are of different characteristics – these difficulties are reflected in the released dataset.

The participants are provided with the AI-ready dataset (RFI dataset), comprising **3,940 quicklook (RGB) images**. The dataset was divided into training and test sets based on multiple criteria to ensure balanced representation across geographic regions, RFI sizes, and types of interference.

The training set contains **3,154 images**, accompanied with the ground-truth annotations (bounding boxes showing RFI artifacts), whereas the test set contains **786 images**. Furthermore, the validation set includes 50% of test images.

The score obtained by a Team is calculated using **mean Average Precision (mAP)**. It is calculated at 10 Intersection-over-Union (IoU) thresholds of .50:.05:.95, and mAP ranges from 0 to 1, with 1 indicating the perfect score. The script used to calculate mAP is released as part of the Starter Pack.