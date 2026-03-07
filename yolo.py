from argparse import ArgumentParser, Namespace
import yaml
from typing import Any
from ultralytics import YOLO


def parse_args() -> tuple[str, str]:
    """
    Parses the command line arguments to retrieve the mode and path to the YAML configuration file.
    :return: A tuple containing the mode and the path to the YAML configuration file.
    """
    parser = ArgumentParser(description="Parse mode and YAML configuration file for running the experiment.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval", "inference"], help="Mode to run the experiment in (train/eval/inference).")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args: Namespace = parser.parse_args()

    return args.mode, args.config


def read_yaml_config(yaml_file: str) -> dict[str, Any]:
    """
    Reads the training configuration from a YAML file.

    :param yaml_file: Path to the YAML file.
    :return: Dictionary of the parsed YAML configuration.
    """
    try:
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: YAML configuration file '{yaml_file}' not found.")
        exit()
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        exit()


def train(config_file_path: str) -> None:

    # Read YAML config file and transform it into a dict
    train_args = read_yaml_config(config_file_path)

    print("PERFORMING TRAINING WITH THE FOLLOWING ARGUMENTS:")
    print(train_args, "\n")

    # Load the model
    model_checkpoint = train_args.pop("model")
    model = YOLO(model=model_checkpoint, task="detect")

    # Train model
    model.train(**train_args)

def eval(config_file_path: str) -> None:

    # Read YAML config file and transform it into a dict
    eval_args = read_yaml_config(config_file_path)

    print("PERFORMING EVALUATION WITH THE FOLLOWING ARGUMENTS:")
    print(eval_args, "\n")

    # Load the model
    model_checkpoint = eval_args.pop("model")
    model = YOLO(model=model_checkpoint, task="detect")

    # Evaluate model
    model.val(**eval_args)


def inference(config_file_path: str) -> None:
    
    # Read YAML config file and transform it into a dict
    inference_args = read_yaml_config(config_file_path)

    print("PERFORMING INFERENCE WITH THE FOLLOWING ARGUMENTS:")
    print(inference_args, "\n")

    # Load the model
    model_checkpoint = inference_args.pop("model")
    model = YOLO(model=model_checkpoint, task="detect")

    # Perform inference with the model
    results = model.predict(**inference_args)

    # ---------------

    # Convert predictions to COCO detection format
    #detections = []
    #for result in results:
    #    # Extract image_id from the filename stem (e.g. "1260.png" → 1260)
    #    image_id = int(Path(result.path).stem)
    #
    #    boxes  = result.boxes.xyxy.cpu()   # [x_min, y_min, x_max, y_max]
    #    scores = result.boxes.conf.cpu()
    #    for box, score in zip(boxes, scores):
    #        x_min, y_min, x_max, y_max = box
    #        # Convert to COCO [x, y, w, h] format
    #        coco_box = [x_min, y_min, x_max - x_min, y_max - y_min]
    #        detections.append({
    #            "image_id": image_id,
    #            "category_id": 1,
    #            "bbox": list(map(float, coco_box)),
    #            "score": float(score),
    #        })
    #
    #with open(output_path, "w") as f:
    #    json.dump(detections, f)
    #
    #print(f"Saved {len(detections)} detections to {output_path}")



def to_subsmission():
    pass


if __name__ == "__main__":
    # Parse the config file from command line
    mode, config_file_path = parse_args()
    if mode == "train":
        train(config_file_path)
    elif mode == "eval":
        eval(config_file_path)
    elif mode == "inference":
        inference(config_file_path) 
