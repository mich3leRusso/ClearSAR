from rfdetr import RFDETRMedium
from yolo import read_yaml_config

# RF-DETR Nano
# RF-DETR Small
# RF-DETR Base
# RF-DETR Medium
# RF-DETR Large

CFG = "/home/simone/myprojects/ClearSAR/configs/rfdetr.yaml"

def main():
    model = RFDETRMedium()
    cfg = read_yaml_config(CFG)

    model.train(**cfg)

if __name__ == "__main__":
    main()