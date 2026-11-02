"""Unit tests for Faster R-CNN detector preprocessing, output schema, metadata validation and error handling."""

import importlib.util
import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Safe loader that allows testing even if cv2 is not yet installed in this python environment
def _load_module(mod_name, file_rel_path):
    path = os.path.join(PROJECT_ROOT, file_rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

try:
    from helmet.label_parser import CLASS_NAMES, NUM_CLASSES
    from helmet.frcnn_detector import (
        FasterRCNNHelmetDetector,
        format_detections,
        get_inference_device,
        preprocess_image,
        validate_checkpoint_metadata,
    )
except ModuleNotFoundError:
    label_parser_mod = _load_module("helmet.label_parser", os.path.join("helmet", "label_parser.py"))
    frcnn_mod = _load_module("helmet.frcnn_detector", os.path.join("helmet", "frcnn_detector.py"))

    CLASS_NAMES = label_parser_mod.CLASS_NAMES
    NUM_CLASSES = label_parser_mod.NUM_CLASSES
    FasterRCNNHelmetDetector = frcnn_mod.FasterRCNNHelmetDetector
    format_detections = frcnn_mod.format_detections
    get_inference_device = frcnn_mod.get_inference_device
    preprocess_image = frcnn_mod.preprocess_image
    validate_checkpoint_metadata = frcnn_mod.validate_checkpoint_metadata

import torch


class TestFasterRCNNDetector(unittest.TestCase):

    def test_01_device_selection(self):
        """Verify device selection supports explicit CPU and auto-detects MPS when available."""
        cpu_dev = get_inference_device("cpu")
        self.assertEqual(cpu_dev.type, "cpu")

        auto_dev = get_inference_device()
        self.assertIn(auto_dev.type, ["mps", "cuda", "cpu"])
        if torch.backends.mps.is_available():
            self.assertEqual(auto_dev.type, "mps")

    def test_02_weights_availability_check(self):
        """Verify detector checks weight existence cleanly and raises FileNotFoundError on missing weights."""
        detector = FasterRCNNHelmetDetector(
            weights_path="/nonexistent/path/frcnn_best.pth",
            lazy_load=True,
        )
        self.assertFalse(detector.is_weights_available())

        # Attempting to predict without weights raises informative FileNotFoundError
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(FileNotFoundError) as ctx:
            detector.predict(dummy_img)
        self.assertIn("checkpoint not found", str(ctx.exception).lower())

    def test_03_preprocessing_numpy_array(self):
        """Verify BGR numpy array converts to [3, H, W] float32 tensor normalized in [0, 1]."""
        # Create 100x200 BGR test image
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Blue channel

        tensor, (h, w) = preprocess_image(img)
        self.assertEqual(h, 100)
        self.assertEqual(w, 200)
        self.assertEqual(tensor.shape, (3, 100, 200))
        self.assertEqual(tensor.dtype, torch.float32)

        # In RGB, Red is channel 0, Blue is channel 2
        # Original Blue (255) should now be in channel 2 with value 1.0
        self.assertAlmostEqual(float(tensor[2, 0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(tensor[0, 0, 0]), 0.0, places=4)

    def test_04_preprocessing_invalid_inputs(self):
        """Verify invalid image shapes or types raise appropriate errors."""
        with self.assertRaises(ValueError):
            preprocess_image(np.zeros((100, 100), dtype=np.uint8))  # 2D grayscale
        with self.assertRaises(TypeError):
            preprocess_image(12345)
        with self.assertRaises(FileNotFoundError):
            preprocess_image("/invalid/path/file.jpg")

    def test_05_format_detections_mock_output(self):
        """Verify format_detections accurately maps model predictions to structured violations."""
        mock_boxes = [[10.0, 20.0, 80.0, 150.0], [50.0, 60.0, 200.0, 250.0]]
        mock_scores = [0.92, 0.40]
        # Class 1: DHelmet (compliant)
        # Class 7: DHelmetP1NoHelmet (violation)
        mock_labels = [1, 7]

        # conf_threshold = 0.5 should keep box 1 and discard box 2
        res = format_detections(
            boxes=mock_boxes,
            scores=mock_scores,
            labels=mock_labels,
            conf_threshold=0.50,
            img_size=(300, 300),
        )
        self.assertEqual(len(res), 1)
        item = res[0]
        self.assertEqual(item["box"], [10, 20, 80, 150])
        self.assertEqual(item["class_id"], 1)
        self.assertEqual(item["label"], "DHelmet")
        self.assertEqual(item["rider_count"], 1)
        self.assertTrue(item["driver_has_helmet"])
        self.assertFalse(item["is_violation"])

        # conf_threshold = 0.35 should keep both boxes
        res_both = format_detections(
            boxes=mock_boxes,
            scores=mock_scores,
            labels=mock_labels,
            conf_threshold=0.35,
            img_size=(300, 300),
        )
        self.assertEqual(len(res_both), 2)
        v_item = res_both[1]
        self.assertEqual(v_item["label"], "DHelmetP1NoHelmet")
        self.assertEqual(v_item["rider_count"], 2)
        self.assertTrue(v_item["has_no_helmet_violation"])
        self.assertTrue(v_item["is_violation"])
        self.assertIn("Passenger P1 without helmet", v_item["violation_reasons"])

    def test_06_metadata_validation(self):
        """Verify checkpoint metadata validation checks num_classes == 17 and class names."""
        # Valid checkpoint dictionary
        valid_checkpoint = {
            "num_classes": NUM_CLASSES,  # 17
            "class_names": list(CLASS_NAMES),
            "model_state_dict": {},
        }
        # Should pass without error
        validate_checkpoint_metadata(valid_checkpoint)

        # Mismatched class count should raise ValueError
        invalid_count = {
            "num_classes": 10,
            "class_names": list(CLASS_NAMES),
        }
        with self.assertRaises(ValueError) as ctx:
            validate_checkpoint_metadata(invalid_count)
        self.assertIn("class count mismatch", str(ctx.exception).lower())

        # Mismatched class names should raise ValueError
        invalid_names = {
            "num_classes": NUM_CLASSES,
            "class_names": ["WrongName"] * 16,
        }
        with self.assertRaises(ValueError) as ctx:
            validate_checkpoint_metadata(invalid_names)
        self.assertIn("class names do not match", str(ctx.exception).lower())

    def test_07_mps_fallback_support(self):
        """Verify device selection gracefully handles explicit devices and probes."""
        dev = get_inference_device("cpu")
        self.assertEqual(dev.type, "cpu")


if __name__ == "__main__":
    unittest.main()
