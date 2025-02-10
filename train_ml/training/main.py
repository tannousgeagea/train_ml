# from ultralytics import YOLO

# model = YOLO("/home/appuser/src/train_ml/runs/detect/amk_front_impurity.v2/weights/best.pt")


# model.train(
#     data='/media/amk_front_impurity.v1/data.yaml',
#     imgsz=640,
#     epochs=200,
#     batch=16,
#     lr0=0.0001,
#     lrf=0.00001,
#     augment=False,
#     name='amk_front_impurity.v3'
# )


# import cv2
# import os
# import django
# import random
# django.setup()
# import numpy as np
# from typing import Tuple
# from common_utils.detection.convertor import copy_and_paste
# from common_utils.detection.core import Detections
# from common_utils.model.base import BaseModels
# from ml_models.models import (
#     ModelVersion
# )

# seg_model = ModelVersion.objects.filter(
#     model__name="waste-segmentation-gate"
# ).order_by('-version').first()


# seg_model = BaseModels(
#     weights=seg_model.checkpoint.path,
# )

# model = BaseModels(
#     weights='/home/appuser/src/train_ml/runs/detect/waste_material_classification_synthetic/weights/best.pt'
#     )


# source = "/media/amk_front_impurity.v5i.yolov8/train/images/"
# output_dir = "./output"
# os.makedirs(output_dir, exist_ok=True)

# images = os.listdir(source)
# for image in images:
#     cv_image = cv2.imread(
#         os.path.join(source, image)
#     )

#     detections = Detections.from_dict({})
#     results = seg_model.classify_one(
#         image=cv_image,
#         conf=0.25,
#         mode="detect"
#     )
#     detections = detections.from_dict(results)
#     xy = detections.xy
#     xyxy = detections.xyxyn


#     classes = []
#     class_ids = []
#     confidences = []
#     filename = os.path.basename(image).split('.')[0]
#     for i, polygon in enumerate(xy):
#         background = copy_and_paste(
#             img=cv_image,
#             polygon=polygon
#             )
#         res = model.classify_one(
#             background
#         )
        
#         detections = Detections.from_dict(res)
#         if len(detections):
#             cls_id = int(detections.class_id[0])
#             class_ids.append(cls_id)
#             conf = float(detections.confidence[0])
#             class_name = res.get("class_names")[cls_id]
#             classes.append(res.get("class_names")[cls_id])
#             confidences.append(float(detections.confidence[0]))
#         else:
#             class_ids.append(10)
#             class_name = "unknown"
#             classes.append('unknown')
#             confidences.append(0)
#             conf = 0

#         out_dir = output_dir + "/" + class_name
#         os.makedirs(out_dir, exist_ok=True)
#         cv2.putText(background, f"{class_name} {conf:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA, False)
#         cv2.imwrite(f"{out_dir}/{filename}_{str(i)}.png", background)

import os
import cv2
import django
django.setup()
import numpy as np
from common_utils.annotation.utils import load_yolo_segmentation_labels
from common_utils.image.utils import (
    read_image,
    get_all_files,
    load_image_and_mask
)

from common_utils.model.base import BaseModels
from common_utils.detection.core import Detections
from common_utils.detection.convertor import (
    copy_and_paste
)

from ml_models.models import (
    ModelVersion
)

model = BaseModels(
    weights=ModelVersion.objects.filter(
        model__name="waste-material-classification"
    ).order_by('-version').first().checkpoint.path
)

classes = ['cardboard', 'fabric', 'metal', 'plastic', 'rubber', 'wood']
source = "/media/waste_material_classification.v1i.yolov8/train"
output_dir = "./output"

images = get_all_files(
    os.path.join(source, "images")
)
print(len(images))
print('hello')

fp = 0
fn = 0
tp =  0

target_class = 'plastic'

for image_path in images:
    image, polygons = load_image_and_mask(
        image_path=image_path, 
        annotation_dir=os.path.join(source, "labels"),
        annotation_mode="seg",
    )
    
    for i, polygon in enumerate(polygons):
        class_id = polygon['class_id']
        xy = np.array((polygon['xy'])).astype(np.int32)
        
        # if classes[class_id] != "plastic":
        #     continue
        
        background = copy_and_paste(
            img=image,
            polygon=xy,
            kernel=np.ones((5, 5), np.uint8)
        )
        
        filename = os.path.basename(image_path).split('.')[0]
        out_dir = output_dir
        cv2.imwrite(f"{out_dir}/{filename}_{str(i)}.png", background)
        
        continue
        res = model.classify_one(
            background, conf=0.6
        )
        
        detections = Detections.from_dict(res)
    
        if not len(detections):
            continue
        
        if len(detections):
            cls_id = int(detections.class_id[0])
            conf = float(detections.confidence[0])
            class_name = res.get("class_names")[cls_id]
            classes.append(res.get("class_names")[cls_id])
        else:
            class_name = "unknown"
            classes.append('unknown')
            conf = 0
        
        if classes[class_id] == "plastic":
            if class_name == "PLASTIC":
                tp += 1
                out = 'tp'
            else:
                fp += 1
                out = 'fp'
        else:
            if class_name == "PLASTIC":
                fn += 1
                out = 'fn'
            else:
                out = 'tn'
        
        filename = os.path.basename(image_path).split('.')[0]
        out_dir = output_dir + "/" + out
        os.makedirs(out_dir, exist_ok=True)
        cv2.putText(background, f"{classes[class_id]} vs {class_name} {conf:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA, False)
        cv2.imwrite(f"{out_dir}/{filename}_{str(i)}.png", background)
        
print("TP: ", tp)
print("FP: ", fp)
print("FN: ", fn)