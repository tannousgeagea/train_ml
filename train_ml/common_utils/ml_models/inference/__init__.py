from .yolo.predictor import YOLOInferencePlugin

INFERENCE_PLUGIN = {
    "yolo": YOLOInferencePlugin
}

def get_inference_plugin(framework:str):
    predictor = INFERENCE_PLUGIN.get(framework)
    if not predictor:
        raise ValueError(f"No inference plugin for model type: {framework}")

    return predictor
