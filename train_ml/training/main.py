from ultralytics import YOLO

model = YOLO("yolo11s.pt")


model.train(
    data='/media/GARBAGE_CLASSIFICATION_3.v2-gc1.yolov8/data.yaml',
    imgsz=416,
    epochs=200,
    batch=16,
    lr0=0.001,
    lrf=0.00001,
    augment=False,
    name='garbage_classification.v1.2'
)

