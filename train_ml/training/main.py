from ultralytics import YOLO

model = YOLO("./runs/detect/garbage_classification.v1.2/weights/best.pt")


model.train(
    data='/media/Waste_Material_Classification_Synthetic/data.yaml',
    imgsz=416,
    epochs=200,
    batch=16,
    lr0=0.001,
    lrf=0.00001,
    augment=False,
    name='waste_material_classification_synthetic'
)

