import os
import sys
import time
import uuid
from typing import Optional

import cv2
import numpy as np
from sqlalchemy.orm import Session

# Add project root directory to path to access helmet and number_plate modules
ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.models import Violation

EVIDENCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "evidence",
)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Path to the primary trained Faster R-CNN checkpoint
FRCNN_CHECKPOINT_PATH = os.path.join(
    ROOT_DIR, "checkpoints", "frcnn_16class_best.pth"
)


# ---------------------------------------------------------------------------
# Model Inference Interface / Adapters
# ---------------------------------------------------------------------------

class BaseHelmetDetector:
    """Base interface for helmet detection models."""

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> list[dict]:
        raise NotImplementedError


class MockHelmetDetector(BaseHelmetDetector):
    """Simulated detector for testing backend endpoints, upload validation,

    and database storage without falsely presenting results as genuine AI predictions.
    """

    def __init__(self, test_detections: Optional[list[dict]] = None):
        self.test_detections = test_detections or []

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> list[dict]:
        """Returns an empty helmet detection list in mock mode unless valid test coordinates

        are explicitly supplied. Never implies a fabricated location is a genuine detection.
        """
        return list(self.test_detections)


class FasterRCNNLiveDetector(BaseHelmetDetector):
    """Executes helmet detection using trained Faster R-CNN 16-class ResNet-50 FPN model."""

    def __init__(self, weights_path: Optional[str] = None):
        from helmet.frcnn_detector import FasterRCNNHelmetDetector
        self.detector = FasterRCNNHelmetDetector(
            weights_path=weights_path or FRCNN_CHECKPOINT_PATH,
            lazy_load=True,
        )

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> list[dict]:
        return self.detector.predict(image, conf_threshold=conf_threshold)


def get_helmet_detector(
    force_mock: bool = False,
    test_detections: Optional[list[dict]] = None,
) -> tuple[BaseHelmetDetector, str, bool]:
    """Factory function returning the active detector, status message, and mock flag.

    Faster R-CNN (16-Class ResNet-50 FPN) is the ONLY active model for helmet detection.
    """
    if force_mock:
        return MockHelmetDetector(test_detections=test_detections), "mock_inference_mode", True

    if os.path.isfile(FRCNN_CHECKPOINT_PATH):
        try:
            detector = FasterRCNNLiveDetector(FRCNN_CHECKPOINT_PATH)
            return detector, "faster_rcnn_16class_active", False
        except Exception as err:
            print(f"Faster R-CNN initialization error: {err}")
            return MockHelmetDetector(test_detections=test_detections), "mock_fallback_mode", True

    # Fallback to transparent mock if checkpoint is missing
    return MockHelmetDetector(test_detections=test_detections), "mock_fallback_mode", True


# ---------------------------------------------------------------------------
# Visual Annotation & Pipeline
# ---------------------------------------------------------------------------

def annotate_frame(
    image: np.ndarray,
    helmet_detections: list[dict],
    plate_detections: list[dict],
    is_mock: bool = False,
) -> np.ndarray:
    """Draw bounding boxes and labels for detected helmets and license plates."""
    annotated = image.copy()

    # Draw Faster R-CNN rider & helmet detections
    for item in helmet_detections:
        x1, y1, x2, y2 = item["box"]
        label = item.get("readable_label", item.get("label", "Helmet"))
        conf = item.get("confidence", 0.0)

        # Violation if NoHelmet or overcrowding (>= 3 riders)
        is_violation = (
            item.get("is_violation", False)
            or "nohelmet" in item.get("label", "").lower()
            or "without" in item.get("label", "").lower()
            or item.get("rider_count", 1) >= 3
        )

        color = (0, 0, 220) if is_violation else (0, 210, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        prefix = "[MOCK] " if is_mock else ""
        tag = f"{prefix}{label} ({conf:.2f})"
        cv2.putText(
            annotated,
            tag,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
        )

    # Draw license plate detections (YOLO + EasyOCR)
    for plate in plate_detections:
        x1, y1, x2, y2 = plate["plate_box"]
        plate_text = plate.get("plate_text") or "Plate"
        conf = plate.get("detector_confidence", 0.0)

        color = (255, 200, 0)  # Cyan/Yellow
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        tag = f"Number Plate: {plate_text} ({conf:.2f})"
        cv2.putText(
            annotated,
            tag,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return annotated


def analyze_traffic_image(
    image_bytes: bytes,
    original_filename: str,
    db: Session,
    detection_source: str = "camera_upload",
    conf_threshold: float = 0.2,
    force_mock: bool = False,
    test_detections: Optional[list[dict]] = None,
) -> dict:
    """Execute model inference workflow:

    1. Validates and decodes image bytes.
    2. Runs Faster R-CNN 16-class helmet detector.
    3. Runs license-plate detector + EasyOCR.
    4. Evaluates compliance & violations.
    5. Saves annotated evidence image.
    6. Persists structured record into PostgreSQL.
    """
    start_time = time.time()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid or corrupted image format. Unable to decode image.")

    # 1. Faster R-CNN Helmet & Rider Detection
    detector, model_status, is_mock = get_helmet_detector(
        force_mock=force_mock,
        test_detections=test_detections,
    )
    helmet_detections = detector.predict(image, conf_threshold=conf_threshold)

    # 2. License Plate Detection & OCR (preserved)
    plate_detections = []
    try:
        from number_plate.pipeline import process_traffic_image
        plate_detections = process_traffic_image(image, conf_threshold=conf_threshold)
    except Exception as err:
        # Graceful fallback if OCR is unavailable or fails
        print(f"License plate pipeline notice: {err}")
        plate_detections = []

    # 3. Compliance Decision Logic
    has_violation = any(
        d.get("is_violation", False)
        or "nohelmet" in d.get("label", "").lower()
        or "without" in d.get("label", "").lower()
        or d.get("rider_count", 1) >= 3
        for d in helmet_detections
    )
    has_compliant = any(
        not d.get("is_violation", False)
        and ("helmet" in d.get("label", "").lower() or "with" in d.get("label", "").lower())
        and "without" not in d.get("label", "").lower()
        and "nohelmet" not in d.get("label", "").lower()
        for d in helmet_detections
    )

    if has_violation:
        helmet_detected = False
        review_status = "pending"
    elif has_compliant:
        helmet_detected = True
        review_status = "approved"
    else:
        helmet_detected = True
        review_status = "approved"

    # 4. Extract Primary License Plate Text (never invent)
    recognized_plates = [
        p["plate_text"] for p in plate_detections if p.get("plate_text") and p.get("plate_text").strip()
    ]
    primary_plate = recognized_plates[0] if recognized_plates else None

    # 5. Save Annotated Evidence File
    annotated = annotate_frame(image, helmet_detections, plate_detections, is_mock=is_mock)
    unique_suffix = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    evidence_filename = f"evidence_{unique_suffix}.jpg"
    evidence_disk_path = os.path.join(EVIDENCE_DIR, evidence_filename)
    cv2.imwrite(evidence_disk_path, annotated)

    evidence_url_path = f"/evidence/{evidence_filename}"

    # 6. Database Storage in existing PostgreSQL table
    source_tag = f"{detection_source}-mock" if is_mock else detection_source
    violation = Violation(
        number_plate=primary_plate,
        helmet_detected=helmet_detected,
        detection_source=source_tag,
        review_status=review_status,
        evidence_path=evidence_url_path,
    )
    db.add(violation)
    db.commit()
    db.refresh(violation)

    # 7. Summary Message
    reasons = []
    for d in helmet_detections:
        if d.get("violation_reasons"):
            reasons.extend(d["violation_reasons"])
        elif "without" in d.get("label", "").lower() or "nohelmet" in d.get("label", "").lower():
            reasons.append("Rider without helmet")
        elif d.get("rider_count", 1) >= 3:
            reasons.append(f"Illegal overload ({d.get('rider_count')} riders)")

    # Remove duplicates preserving order
    reasons = list(dict.fromkeys(reasons))

    if is_mock:
        if has_violation:
            summary = f"[Simulated Test] Simulated violation ({'; '.join(reasons) or 'no helmet'}). Plate: {primary_plate or 'None'}"
        elif has_compliant:
            summary = f"[Simulated Test] Simulated compliant. Plate: {primary_plate or 'None'}"
        else:
            summary = f"[Simulated Test] Model running in test mode. Plate: {primary_plate or 'None'}"
    elif has_violation:
        reason_str = "; ".join(reasons) if reasons else "Rider without helmet"
        summary = f"Potential violation: {reason_str}. Plate: {primary_plate or 'Unreadable'}"
    elif has_compliant:
        summary = f"Compliant: Rider wearing helmet. Plate: {primary_plate or 'Unreadable'}"
    else:
        summary = f"No riders identified in frame. Plate: {primary_plate or 'None'}"

    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "violation": {
            "id": violation.id,
            "number_plate": violation.number_plate,
            "helmet_detected": violation.helmet_detected,
            "detection_source": violation.detection_source,
            "review_status": violation.review_status,
            "evidence_path": violation.evidence_path,
            "created_at": violation.created_at,
        },
        "helmet_classification": [
            {
                "box": d["box"],
                "label": d.get("label"),
                "readable_label": d.get("readable_label", d.get("label")),
                "confidence": d["confidence"],
                "class_id": d.get("class_id"),
                "rider_count": d.get("rider_count", 1),
                "is_violation": d.get("is_violation", False),
                "violation_reasons": d.get("violation_reasons", []),
            }
            for d in helmet_detections
        ],
        "number_plate_detection": [
            {
                "plate_box": p["plate_box"],
                "confidence": p["detector_confidence"],
                "plate_text": p["plate_text"],
            }
            for p in plate_detections
        ],
        "ocr_output": {
            "primary_plate": primary_plate,
            "all_plates": recognized_plates,
        },
        "evidence_path": evidence_url_path,
        "is_mock_inference": is_mock,
        "model_status": model_status,
        "summary": summary,
        "processing_time_ms": elapsed_ms,
    }
