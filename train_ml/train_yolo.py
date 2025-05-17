import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model with custom parameters")

    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to the data.yaml file')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'], help='Optimizer type')
    parser.add_argument('--lr0', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.00001, help='Final learning rate fraction')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--name', type=str, required=True, help='Experiment name (run directory name)')

    args = parser.parse_args()

    model = YOLO(args.weights)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        epochs=args.epochs,
        batch=args.batch,
        augment=args.augment,
        name=args.name,
    )

if __name__ == "__main__":
    main()
