from .yolo.trainer import YOLOTrainer

TRAINER_MAP = {
    "yolo": YOLOTrainer,
}

def get_trainer(framework_name: str):
    trainer_cls = TRAINER_MAP.get(framework_name.lower())
    if not trainer_cls:
        raise ValueError(f"No trainer found for framework: {framework_name}")
    return trainer_cls
