import time
import torch
from ultralytics.nn.tasks import parse_model, yaml_model_load
from ultralytics.utils.torch_utils import smart_inference_mode

def attempt_load_one_weight(weight_path, device=None, inplace=True, fuse=False):
    """
    Load a YOLO model from a weight (.pt) file.

    Args:
        weight_path (str): Path to the weight file.
        device (str|torch.device): Device to load the model onto.
        inplace (bool): Load model inplace (keep original object).
        fuse (bool): Fuse Conv+BN layers for faster inference.

    Returns:
        torch.nn.Module: Loaded YOLO model.
    """
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)

    # Extract the model
    model = ckpt['model']
    if not inplace:
        model = model.float().fuse().eval() if fuse else model.float().eval()
    else:
        model.float()
        if fuse:
            model.fuse()
        model.eval()

    model.to(device)
    return model


class YOLOBuilder:
    def __init__(self, config_path, checkpoint=None, device='cpu'):
        self.device = torch.device(device)
        self.model = self.build_model(config_path, checkpoint)

    def build_model(self, config_path, checkpoint=None):
        if checkpoint:
            model = attempt_load_one_weight(checkpoint, device=self.device)
        else:
            model_cfg = yaml_model_load(config_path)
            model = parse_model(model_cfg, ch=[3])[0]
            model.to(self.device)
        return model


    def summary(self, input_shape=(1, 3, 640, 640)):
        from torchinfo import summary
        return summary(self.model, input_size=input_shape, col_names=["input_size", "output_size", "num_params"])

    @smart_inference_mode()
    def test_forward(self, image_tensor):
        print("\n🚀 Running Inference:")
        start = time.time()
        outputs = self.model(image_tensor)
        end = time.time()

        print(f"Inference time: {end - start:.3f}s")
        if isinstance(outputs, (list, tuple)):
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    print(f"Output[{i}]: Tensor shape: {out.shape}")
                elif isinstance(out, list):
                    print(f"Output[{i}]: List with {len(out)} elements")
                else:
                    print(f"Output[{i}]: {type(out)}")
        else:
            print(f"Output: {type(outputs)}")
        return outputs


import os
from ultralytics.models.yolo.detect.train import DetectionTrainer as Trainer
from ultralytics.utils import LOGGER
from ultralytics.cfg import get_cfg
from ultralytics.utils import downloads

def no_download_asset(path, *args, **kwargs):
    print(f"❌ [BLOCKED] attempt_download_asset('{path}')")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found locally: {path}")
    return path

downloads.attempt_download_asset = no_download_asset

class CustomTrainer(Trainer):
    def __init__(self, overrides):
        self.local_model_path = overrides.pop("local_model_path", None)
        if not self.local_model_path:
            raise ValueError("Missing 'local_model_path' in overrides")
         # prevents fallback to hub defaults
        args = get_cfg(overrides=overrides)  # apply CLI-like overrides
        super().__init__(args)

    # def get_model(self, cfg=None, weights=None, verbose=True):
    #     """
    #     Overrides the default get_model method to load a local checkpoint instead of downloading.
    #     """
    #     if self.local_model_path:

    #         weights = None
    #         print(f"📦 Loading local model from: {self.local_model_path}")
    #         ckpt = torch.load(self.local_model_path, map_location=self.device, weights_only=False)
    #         model = ckpt["model"].float()
    #         model.to(self.device)
    #         model.args = self.args  # required by trainer

    #         # ✅ Unfreeze all layers manually before training
    #         for p in model.parameters():
    #             p.requires_grad = True
    #         model.frozen = False 
    #         return model
    #     else:
    #         raise ValueError("Missing 'local_model_path' in overrides")

    # def setup_model(self):
    #     # Override default model setup to guarantee it's taken from get_model only
    #     # if self.model is None:
    #     self.model = self.get_model(cfg=self.args.model, weights=None)
    #     return None

training_config = {
    'local_model_path': '/media/models/WasteImpurityMultiClass_agr_bunker_V4.pt',
    'data': '/media/agr_impurity_detection.v5.yolo/data.yaml',                  # path to dataset config
    'epochs': 50,
    'batch': 16,
    'imgsz': 640,
    'device': 0, 
    'project': 'runs/train',
    'name': 'exp1',
    'exist_ok': True,
    'pretrained': True,
    "model": "/media/models/WasteImpurityMultiClass_agr_bunker_V4.pt"
}


trainer = CustomTrainer(overrides=training_config)
trainer.train()

# weight_path="/media/models/WasteImpurityMultiClass_agr_bunker_V4.pt"
# model = YOLOBuilder(checkpoint=weight_path, config_path="")
# dummy_input = torch.randn(1, 3, 640, 640)
# outputs = model.test_forward(dummy_input)

# trainer = YOLOTrainer(config=training_config)
# best_model_path = trainer.train()
# print("Best checkpoint saved at:", best_model_path)