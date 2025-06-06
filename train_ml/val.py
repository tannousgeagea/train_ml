import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model with custom parameters")

    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to the data.yaml file')

    args = parser.parse_args()

    model = YOLO(args.weights)

    model.val(
        data=args.data,
    )

if __name__ == "__main__":
    main()
