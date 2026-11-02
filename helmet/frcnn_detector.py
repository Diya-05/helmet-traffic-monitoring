"""Faster R-CNN 16-Class Helmet and Traffic Violation Detector.

Uses torchvision Faster R-CNN ResNet-50 FPN with 17 outputs (background + 16 classes).
Supports Apple Silicon MPS and CPU fallback, validates checkpoint metadata,
and integrates helmet.label_parser.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from helmet.label_parser import (
        CLASS_NAMES,
        ID_TO_LABEL,
        LABEL_TO_ID,
        NUM_CLASSES,
        NUM_OBJECT_CLASSES,
        parse_compound_label,
    )
except (ImportError, ModuleNotFoundError):
    # Standalone fallback if package root is not yet resolved
    import importlib.util
    _parser_path = os.path.join(os.path.dirname(__file__), "label_parser.py")
    _spec = importlib.util.spec_from_file_location("helmet.label_parser", _parser_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    CLASS_NAMES = _mod.CLASS_NAMES
    ID_TO_LABEL = _mod.ID_TO_LABEL
    LABEL_TO_ID = _mod.LABEL_TO_ID
    NUM_CLASSES = _mod.NUM_CLASSES
    NUM_OBJECT_CLASSES = _mod.NUM_OBJECT_CLASSES
    parse_compound_label = _mod.parse_compound_label

DEFAULT_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "checkpoints", "frcnn_16class_best.pth"
)


def get_inference_device(preferred: Optional[str] = None) -> Any:
    """Determine the optimal compute device with graceful Apple Silicon MPS / CPU fallback.

    Args:
        preferred: Explicit device choice (e.g., 'cpu', 'mps', 'cuda').

    Returns:
        torch.device
    """
    import torch

    if preferred is not None:
        return torch.device(preferred)

    # Attempt Apple Silicon GPU (MPS) first on macOS
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            # Sanity probe to ensure MPS runtime is fully functional
            probe = torch.zeros(1, device="mps")
            del probe
            return torch.device("mps")
        except Exception as err:
            print(f"Notice: MPS probe failed ({err}). Falling back to CPU.")
            return torch.device("cpu")

    # NVIDIA CUDA fallback (e.g. on Linux GPU servers)
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Universal CPU fallback
    return torch.device("cpu")


def build_frcnn_architecture(num_classes: int = NUM_CLASSES) -> Any:
    """Build Faster R-CNN ResNet-50 FPN model with 17 output classes (1 background + 16 classes).

    Matches the exact architecture specified in training/model.py.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=720,
        max_size=1280,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def preprocess_image(
    image_input: Union[str, np.ndarray]
) -> Tuple[Any, Tuple[int, int]]:
    """Convert an image path or OpenCV BGR numpy array into an RGB float tensor [3, H, W] in [0, 1].

    Returns:
        (image_tensor, (height, width))
    """
    import torch

    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image file does not exist: {image_input}")
        try:
            import cv2
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise ValueError(f"Could not read image: {image_input}")
        except ImportError:
            from PIL import Image
            pil_img = Image.open(image_input).convert("RGB")
            rgb = np.array(pil_img)
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
            return tensor, pil_img.size[::-1]
    elif isinstance(image_input, np.ndarray):
        bgr = image_input
    else:
        raise TypeError(f"Expected file path or np.ndarray, got {type(image_input).__name__}")

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image (H, W, 3), got shape {bgr.shape}")

    orig_h, orig_w = bgr.shape[:2]

    # Convert OpenCV BGR to RGB
    try:
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        rgb = bgr[:, :, ::-1]  # BGR to RGB reverse slicing fallback

    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    return tensor, (orig_h, orig_w)


def validate_checkpoint_metadata(checkpoint: Any) -> None:
    """Validate that the checkpoint dictionary matches the expected 17-class dataset metadata.

    Args:
        checkpoint: Loaded torch checkpoint dictionary or state dict.
    """
    if not isinstance(checkpoint, dict):
        return

    # Check num_classes if present in metadata
    if "num_classes" in checkpoint:
        actual_num = checkpoint["num_classes"]
        if actual_num != NUM_CLASSES:
            raise ValueError(
                f"Checkpoint class count mismatch: expected {NUM_CLASSES} "
                f"(1 background + {NUM_OBJECT_CLASSES} classes), got {actual_num}."
            )

    # Check class_names list if present in metadata
    if "class_names" in checkpoint:
        actual_names = checkpoint["class_names"]
        if actual_names != CLASS_NAMES:
            raise ValueError(
                "Checkpoint class names do not match the expected 16 dataset classes. "
                "Class order or naming in checkpoint is inconsistent with training/helmet_dataset.py."
            )


def format_detections(
    boxes: Any,
    scores: Any,
    labels: Any,
    conf_threshold: float = 0.25,
    img_size: Optional[Tuple[int, int]] = None,
    iou_threshold: float = 0.60,
) -> List[Dict[str, Any]]:
    """Filter raw model predictions, apply class-agnostic NMS to eliminate duplicate overlapping
    boxes on the same rider, and augment each detection with compound class violation data.
    """
    import torch
    import torchvision.ops as ops

    if len(boxes) == 0:
        return []

    # Convert to torch tensors if needed
    t_boxes = torch.as_tensor(boxes, dtype=torch.float32)
    t_scores = torch.as_tensor(scores, dtype=torch.float32)

    # Filter by confidence threshold first
    valid_mask = t_scores >= conf_threshold
    if not bool(valid_mask.any()):
        return []

    filtered_boxes = t_boxes[valid_mask]
    filtered_scores = t_scores[valid_mask]
    filtered_labels = [label for idx, label in enumerate(labels) if bool(valid_mask[idx])]

    # Apply class-agnostic NMS to remove duplicate/competing predictions on identical riders
    keep_indices = ops.nms(filtered_boxes, filtered_scores, iou_threshold=iou_threshold)

    results: List[Dict[str, Any]] = []
    orig_h, orig_w = img_size if img_size else (None, None)

    for idx in keep_indices:
        i = int(idx)
        box = filtered_boxes[i]
        conf = float(filtered_scores[i])
        cid = int(filtered_labels[i])

        # Class 0 is background; ignore background predictions
        if cid == 0 or cid > NUM_OBJECT_CLASSES:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]

        # Clamp box to image boundary if dimensions provided
        if orig_w is not None and orig_h is not None:
            x1 = max(0.0, min(x1, float(orig_w)))
            y1 = max(0.0, min(y1, float(orig_h)))
            x2 = max(0.0, min(x2, float(orig_w)))
            y2 = max(0.0, min(y2, float(orig_h)))

        int_box = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        if int_box[2] <= int_box[0] or int_box[3] <= int_box[1]:
            continue

        # Parse compound class (driver, passengers, triple riding, helmet violations)
        parsed = parse_compound_label(cid)

        results.append({
            "box": int_box,
            "class_id": cid,
            "label": parsed.raw_label,
            "confidence": round(conf, 3),
            "readable_label": parsed.readable_label,
            "rider_count": parsed.rider_count,
            "driver_has_helmet": parsed.driver_has_helmet,
            "passengers": parsed.passengers,
            "has_no_helmet_violation": parsed.has_no_helmet_violation,
            "has_driver_violation": parsed.has_driver_violation,
            "has_passenger_violation": parsed.has_passenger_violation,
            "is_triple_riding": parsed.is_triple_riding,
            "is_quad_riding": parsed.is_quad_riding,
            "is_overcrowded": parsed.is_overcrowded,
            "is_violation": parsed.is_violation,
            "violation_reasons": parsed.violation_reasons,
        })

    return results


class FasterRCNNHelmetDetector:
    """Inference engine for 16-class Faster R-CNN helmet and traffic violation detection."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        lazy_load: bool = True,
    ):
        self.weights_path = weights_path or DEFAULT_CHECKPOINT_PATH
        self.device = get_inference_device(device)
        self.model: Optional[Any] = None

        if not lazy_load:
            self.load_model()

    def is_weights_available(self) -> bool:
        """Check whether the trained .pth checkpoint file exists locally."""
        return os.path.isfile(self.weights_path)

    def load_model(self) -> None:
        """Instantiate the architecture and load trained weights onto target device."""
        import torch

        if not self.is_weights_available():
            raise FileNotFoundError(
                f"Faster R-CNN checkpoint not found at: '{self.weights_path}'. "
                f"Please ensure 'frcnn_16class_best.pth' is present in the checkpoints directory."
            )

        print(f"Loading Faster R-CNN weights from {self.weights_path} onto {self.device}...")
        self.model = build_frcnn_architecture(num_classes=NUM_CLASSES)

        checkpoint = torch.load(
            self.weights_path,
            map_location=self.device,
            weights_only=False,
        )

        # Validate checkpoint metadata against Phase 1 specifications
        validate_checkpoint_metadata(checkpoint)

        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Faster R-CNN model ready for inference.")

    def predict(
        self,
        image_input: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Run Faster R-CNN detection on an image with Apple Silicon MPS runtime fallback.

        Args:
            image_input: File path (str) or loaded BGR image (np.ndarray).
            conf_threshold: Minimum confidence score (0.0 to 1.0).

        Returns:
            List of detection dicts with bounding boxes, scores, and parsed violations.
        """
        import torch

        if self.model is None:
            self.load_model()

        tensor, (orig_h, orig_w) = preprocess_image(image_input)
        tensor = tensor.to(self.device)

        self.model.eval()

        # Run inference with MPS -> CPU runtime fallback
        try:
            with torch.inference_mode():
                predictions = self.model([tensor])
        except Exception as err:
            if self.device.type == "mps":
                print(f"Warning: MPS runtime error during inference ({err}). Falling back to CPU.")
                self.device = torch.device("cpu")
                self.model.to(self.device)
                tensor = tensor.to(self.device)
                with torch.inference_mode():
                    predictions = self.model([tensor])
            else:
                raise

        pred = predictions[0]
        boxes = pred["boxes"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
        labels = pred["labels"].detach().cpu().numpy()

        return format_detections(
            boxes=boxes,
            scores=scores,
            labels=labels,
            conf_threshold=conf_threshold,
            img_size=(orig_h, orig_w),
        )

