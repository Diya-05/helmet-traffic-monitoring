import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from training.helmet_dataset import NUM_CLASSES


PROJECT_ROOT = "/mnt/ssd/helmet_project"

PRETRAINED_WEIGHTS = (
    f"{PROJECT_ROOT}/checkpoints/"
    "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
)


def build_model(device=None):
    """
    Build a 16-class Faster R-CNN model.

    Class 0: background
    Classes 1-16: original dataset categories
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if not torch.cuda.is_available() and device.type == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable.")

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=720,
        max_size=1280,
    )

    # Replace the COCO predictor with our 17-class predictor.
    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES,
    )

    # Load COCO pretrained weights, excluding the old 91-class head.
    checkpoint = torch.load(
        PRETRAINED_WEIGHTS,
        map_location="cpu",
        weights_only=True,
    )

    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    filtered_weights = {
        key: value
        for key, value in checkpoint.items()
        if not key.startswith("roi_heads.box_predictor.")
    }

    result = model.load_state_dict(
        filtered_weights,
        strict=False,
    )

    expected_missing = {
        "roi_heads.box_predictor.cls_score.weight",
        "roi_heads.box_predictor.cls_score.bias",
        "roi_heads.box_predictor.bbox_pred.weight",
        "roi_heads.box_predictor.bbox_pred.bias",
    }

    if set(result.missing_keys) != expected_missing:
        raise RuntimeError(
            "Unexpected missing model weights: "
            f"{result.missing_keys}"
        )

    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected pretrained weight keys: "
            f"{result.unexpected_keys}"
        )

    model.to(device)

    return model
