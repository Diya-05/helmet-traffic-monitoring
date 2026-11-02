"""Compound class parser for 16-class Faster R-CNN helmet and rider detection.

Preserves exact class names and 1-based indexing from training/helmet_dataset.py.
"""

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Union

# Exact 16 object class names from training/helmet_dataset.py
# Background is class 0; object classes are 1 through 16.
CLASS_NAMES: List[str] = [
    "DHelmet",                                # 1
    "DHelmetP0NoHelmet",                      # 2
    "DHelmetP0NoHelmetP1NoHelmet",            # 3
    "DHelmetP0NoHelmetP1NoHelmetP2Helmet",    # 4
    "DHelmetP1Helmet",                        # 5
    "DHelmetP1HelmetP2Helmet",                # 6
    "DHelmetP1NoHelmet",                      # 7
    "DHelmetP1NoHelmetP2Helmet",              # 8
    "DHelmetP1NoHelmetP2NoHelmet",            # 9
    "DNoHelmet",                              # 10
    "DNoHelmetP0NoHelmet",                    # 11
    "DNoHelmetP0NoHelmetP1NoHelmet",          # 12
    "DNoHelmetP0NoHelmetP1NoHelmetP2NoHelmet",# 13
    "DNoHelmetP1Helmet",                      # 14
    "DNoHelmetP1NoHelmet",                    # 15
    "DNoHelmetP1NoHelmetP2NoHelmet",          # 16
]

NUM_OBJECT_CLASSES: int = len(CLASS_NAMES)  # 16
NUM_CLASSES: int = NUM_OBJECT_CLASSES + 1   # 17 (includes background class 0)

LABEL_TO_ID: Dict[str, int] = {
    name: index + 1 for index, name in enumerate(CLASS_NAMES)
}

ID_TO_LABEL: Dict[int, str] = {
    index: name for name, index in LABEL_TO_ID.items()
}

# Regex to capture passenger tokens like 'P0NoHelmet', 'P1Helmet', 'P2NoHelmet'
PASSENGER_PATTERN = re.compile(r"(P\d)(Helmet|NoHelmet)")


@dataclass(frozen=True)
class PassengerStatus:
    passenger_id: str   # e.g., 'P0', 'P1', 'P2'
    has_helmet: bool    # True if 'Helmet', False if 'NoHelmet'
    status: str         # 'Helmet' or 'NoHelmet'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedRiderClassification:
    raw_label: str
    class_id: int
    rider_count: int
    driver_has_helmet: bool
    passengers: List[Dict[str, Any]]
    has_no_helmet_violation: bool
    has_driver_violation: bool
    has_passenger_violation: bool
    is_triple_riding: bool
    is_quad_riding: bool
    is_overcrowded: bool
    is_violation: bool
    violation_reasons: List[str]
    readable_label: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_compound_label(
    label_or_id: Union[str, int]
) -> ParsedRiderClassification:
    """Parse a compound class name or class ID into structured traffic compliance data.

    Args:
        label_or_id: The string class name (e.g., 'DHelmetP1NoHelmet')
                     or integer class ID (1-16).

    Returns:
        ParsedRiderClassification with parsed rider count, helmet status,
        overcrowding flags, and violation details.
    """
    if isinstance(label_or_id, int):
        if label_or_id == 0:
            raw_label = "Background"
            class_id = 0
        elif label_or_id in ID_TO_LABEL:
            class_id = label_or_id
            raw_label = ID_TO_LABEL[label_or_id]
        else:
            raise ValueError(
                f"Unknown class index {label_or_id}. Expected 1-{NUM_OBJECT_CLASSES}."
            )
    elif isinstance(label_or_id, str):
        raw_label = label_or_id.strip()
        if raw_label == "Background":
            class_id = 0
        elif raw_label in LABEL_TO_ID:
            class_id = LABEL_TO_ID[raw_label]
        else:
            raise ValueError(
                f"Unknown class name '{raw_label}'. Not in dataset CLASS_NAMES."
            )
    else:
        raise TypeError(f"Expected str or int, got {type(label_or_id).__name__}")

    # Handle background class
    if class_id == 0:
        return ParsedRiderClassification(
            raw_label="Background",
            class_id=0,
            rider_count=0,
            driver_has_helmet=False,
            passengers=[],
            has_no_helmet_violation=False,
            has_driver_violation=False,
            has_passenger_violation=False,
            is_triple_riding=False,
            is_quad_riding=False,
            is_overcrowded=False,
            is_violation=False,
            violation_reasons=[],
            readable_label="Background",
        )

    # 1. Driver status
    if raw_label.startswith("DHelmet"):
        driver_has_helmet = True
        passenger_part = raw_label[len("DHelmet"):]
    elif raw_label.startswith("DNoHelmet"):
        driver_has_helmet = False
        passenger_part = raw_label[len("DNoHelmet"):]
    else:
        raise ValueError(f"Label '{raw_label}' does not start with DHelmet or DNoHelmet")

    # 2. Passenger status
    passengers_list: List[Dict[str, Any]] = []
    for match in PASSENGER_PATTERN.finditer(passenger_part):
        p_id, p_helmet_str = match.groups()
        has_helmet = (p_helmet_str == "Helmet")
        passengers_list.append(
            PassengerStatus(
                passenger_id=p_id,
                has_helmet=has_helmet,
                status=p_helmet_str,
            ).to_dict()
        )

    # 3. Rider counts & Overcrowding
    rider_count = 1 + len(passengers_list)
    is_triple_riding = (rider_count == 3)
    is_quad_riding = (rider_count >= 4)
    is_overcrowded = (rider_count >= 3)

    # 4. Violations
    has_driver_violation = not driver_has_helmet
    has_passenger_violation = any(not p["has_helmet"] for p in passengers_list)
    has_no_helmet_violation = has_driver_violation or has_passenger_violation
    is_violation = has_no_helmet_violation or is_overcrowded

    # 5. Descriptive violation reasons
    violation_reasons: List[str] = []
    if has_driver_violation:
        violation_reasons.append("Driver without helmet")
    for p in passengers_list:
        if not p["has_helmet"]:
            violation_reasons.append(f"Passenger {p['passenger_id']} without helmet")
    if is_triple_riding:
        violation_reasons.append("Illegal triple riding (3 persons on two-wheeler)")
    elif is_quad_riding:
        violation_reasons.append(f"Illegal overload ({rider_count} persons on two-wheeler)")

    # 6. Readable label
    readable_parts = [f"Driver: {'Helmet' if driver_has_helmet else 'No Helmet'}"]
    for p in passengers_list:
        readable_parts.append(f"{p['passenger_id']}: {p['status']}")
    if is_triple_riding:
        readable_parts.append("[Triple Riding]")
    elif is_quad_riding:
        readable_parts.append("[Quad Riding]")

    readable_label = " | ".join(readable_parts)

    return ParsedRiderClassification(
        raw_label=raw_label,
        class_id=class_id,
        rider_count=rider_count,
        driver_has_helmet=driver_has_helmet,
        passengers=passengers_list,
        has_no_helmet_violation=has_no_helmet_violation,
        has_driver_violation=has_driver_violation,
        has_passenger_violation=has_passenger_violation,
        is_triple_riding=is_triple_riding,
        is_quad_riding=is_quad_riding,
        is_overcrowded=is_overcrowded,
        is_violation=is_violation,
        violation_reasons=violation_reasons,
        readable_label=readable_label,
    )

