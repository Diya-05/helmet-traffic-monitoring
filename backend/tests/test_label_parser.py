"""Unit tests verifying compound class parsing across all 16 Faster R-CNN classes."""

import importlib.util
import os
import sys
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
except ModuleNotFoundError:
    # If cv2/dependencies of helmet.detector are not yet installed in this environment,
    # load label_parser directly so unit tests can run independently
    spec = importlib.util.spec_from_file_location(
        "helmet.label_parser",
        os.path.join(PROJECT_ROOT, "helmet", "label_parser.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["helmet.label_parser"] = module
    spec.loader.exec_module(module)

    CLASS_NAMES = module.CLASS_NAMES
    ID_TO_LABEL = module.ID_TO_LABEL
    LABEL_TO_ID = module.LABEL_TO_ID
    NUM_CLASSES = module.NUM_CLASSES
    NUM_OBJECT_CLASSES = module.NUM_OBJECT_CLASSES
    parse_compound_label = module.parse_compound_label


class TestCompoundClassParser(unittest.TestCase):

    def test_class_constants(self):
        """Verify class counts and bijective mappings match dataset definition."""
        self.assertEqual(NUM_OBJECT_CLASSES, 16)
        self.assertEqual(NUM_CLASSES, 17)
        self.assertEqual(len(LABEL_TO_ID), 16)
        self.assertEqual(len(ID_TO_LABEL), 16)

        for name, cid in LABEL_TO_ID.items():
            self.assertEqual(ID_TO_LABEL[cid], name)
            self.assertTrue(1 <= cid <= 16)

    def test_all_16_classes_coverage(self):
        """Ensure every single one of the 16 classes parses successfully without error."""
        for cid in range(1, 17):
            result = parse_compound_label(cid)
            self.assertEqual(result.class_id, cid)
            self.assertEqual(result.raw_label, CLASS_NAMES[cid - 1])

            # String input matches int input
            str_result = parse_compound_label(result.raw_label)
            self.assertEqual(result, str_result)

    def test_solo_driver_compliant(self):
        """Class 1: DHelmet (1 person, compliant)."""
        res = parse_compound_label("DHelmet")
        self.assertEqual(res.rider_count, 1)
        self.assertTrue(res.driver_has_helmet)
        self.assertEqual(len(res.passengers), 0)
        self.assertFalse(res.has_no_helmet_violation)
        self.assertFalse(res.is_overcrowded)
        self.assertFalse(res.is_violation)

    def test_solo_driver_violation(self):
        """Class 10: DNoHelmet (1 person, driver bareheaded)."""
        res = parse_compound_label("DNoHelmet")
        self.assertEqual(res.rider_count, 1)
        self.assertFalse(res.driver_has_helmet)
        self.assertEqual(len(res.passengers), 0)
        self.assertTrue(res.has_no_helmet_violation)
        self.assertTrue(res.has_driver_violation)
        self.assertFalse(res.has_passenger_violation)
        self.assertTrue(res.is_violation)
        self.assertIn("Driver without helmet", res.violation_reasons)

    def test_two_riders_both_compliant(self):
        """Class 5: DHelmetP1Helmet (2 persons, compliant)."""
        res = parse_compound_label("DHelmetP1Helmet")
        self.assertEqual(res.rider_count, 2)
        self.assertTrue(res.driver_has_helmet)
        self.assertEqual(len(res.passengers), 1)
        self.assertTrue(res.passengers[0]["has_helmet"])
        self.assertFalse(res.has_no_helmet_violation)
        self.assertFalse(res.is_violation)

    def test_two_riders_passenger_violation(self):
        """Class 7: DHelmetP1NoHelmet (2 persons, passenger bareheaded)."""
        res = parse_compound_label("DHelmetP1NoHelmet")
        self.assertEqual(res.rider_count, 2)
        self.assertTrue(res.driver_has_helmet)
        self.assertFalse(res.has_driver_violation)
        self.assertTrue(res.has_passenger_violation)
        self.assertTrue(res.has_no_helmet_violation)
        self.assertTrue(res.is_violation)
        self.assertIn("Passenger P1 without helmet", res.violation_reasons)

    def test_two_riders_passenger_p0_violation(self):
        """Class 2: DHelmetP0NoHelmet (2 persons, passenger P0 bareheaded)."""
        res = parse_compound_label("DHelmetP0NoHelmet")
        self.assertEqual(res.rider_count, 2)
        self.assertTrue(res.driver_has_helmet)
        self.assertFalse(res.has_driver_violation)
        self.assertTrue(res.has_passenger_violation)
        self.assertEqual(res.passengers[0]["passenger_id"], "P0")
        self.assertFalse(res.passengers[0]["has_helmet"])
        self.assertTrue(res.is_violation)
        self.assertIn("Passenger P0 without helmet", res.violation_reasons)

    def test_triple_riding_compliant_helmets(self):
        """Class 6: DHelmetP1HelmetP2Helmet (3 persons, helmets on, but triple riding)."""
        res = parse_compound_label("DHelmetP1HelmetP2Helmet")
        self.assertEqual(res.rider_count, 3)
        self.assertTrue(res.is_triple_riding)
        self.assertTrue(res.is_overcrowded)
        self.assertFalse(res.has_no_helmet_violation)
        self.assertTrue(res.is_violation)  # Overcrowding is a traffic violation
        self.assertTrue(any("triple riding" in r.lower() for r in res.violation_reasons))

    def test_triple_riding_with_no_helmet(self):
        """Class 9: DHelmetP1NoHelmetP2NoHelmet (3 persons, 2 passengers bareheaded)."""
        res = parse_compound_label("DHelmetP1NoHelmetP2NoHelmet")
        self.assertEqual(res.rider_count, 3)
        self.assertTrue(res.is_triple_riding)
        self.assertTrue(res.has_no_helmet_violation)
        self.assertTrue(res.has_passenger_violation)
        self.assertEqual(len(res.violation_reasons), 3)  # P1, P2, and triple riding

    def test_quad_riding_detection(self):
        """Class 4 and Class 13 (4 persons on bike)."""
        res4 = parse_compound_label("DHelmetP0NoHelmetP1NoHelmetP2Helmet")
        self.assertEqual(res4.rider_count, 4)
        self.assertTrue(res4.is_quad_riding)
        self.assertTrue(res4.is_overcrowded)
        self.assertTrue(res4.is_violation)

        res13 = parse_compound_label("DNoHelmetP0NoHelmetP1NoHelmetP2NoHelmet")
        self.assertEqual(res13.rider_count, 4)
        self.assertTrue(res13.is_quad_riding)
        self.assertTrue(res13.has_driver_violation)
        self.assertTrue(res13.has_passenger_violation)
        self.assertEqual(len(res13.violation_reasons), 5)  # Driver, P0, P1, P2, and quad overload

    def test_to_dict_serializable(self):
        """Verify .to_dict() returns a clean dictionary suitable for FastAPI responses."""
        res = parse_compound_label("DHelmetP1NoHelmet").to_dict()
        self.assertIsInstance(res, dict)
        self.assertEqual(res["rider_count"], 2)
        self.assertEqual(res["has_no_helmet_violation"], True)

    def test_background_class(self):
        """Verify class 0 / Background handling."""
        res0 = parse_compound_label(0)
        self.assertEqual(res0.class_id, 0)
        self.assertEqual(res0.raw_label, "Background")
        self.assertFalse(res0.is_violation)

        res_bg = parse_compound_label("Background")
        self.assertEqual(res_bg.class_id, 0)
        self.assertEqual(res_bg, res0)

    def test_invalid_input(self):
        """Invalid inputs should raise clear descriptive errors."""
        with self.assertRaises(ValueError):
            parse_compound_label("NonExistentClass")
        with self.assertRaises(ValueError):
            parse_compound_label(99)
        with self.assertRaises(TypeError):
            parse_compound_label(None)


if __name__ == "__main__":
    unittest.main()
