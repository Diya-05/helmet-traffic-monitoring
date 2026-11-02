import os
from collections import defaultdict

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = "/mnt/ssd/helmet_project"

DATASET_ROOT = os.path.join(
    PROJECT_ROOT, "dataset/helmet-dataset"
)

IMAGE_ROOT = os.path.join(
    DATASET_ROOT, "images/part_1"
)

ANNOTATION_ROOT = os.path.join(
    DATASET_ROOT, "annotation/annotation"
)

SPLIT_CSV = os.path.join(
    DATASET_ROOT, "data_split.csv"
)

# 16 original annotation classes.
# Background is class 0; object classes are 1 through 16.

CLASS_NAMES = [
    "DHelmet",
    "DHelmetP0NoHelmet",
    "DHelmetP0NoHelmetP1NoHelmet",
    "DHelmetP0NoHelmetP1NoHelmetP2Helmet",
    "DHelmetP1Helmet",
    "DHelmetP1HelmetP2Helmet",
    "DHelmetP1NoHelmet",
    "DHelmetP1NoHelmetP2Helmet",
    "DHelmetP1NoHelmetP2NoHelmet",
    "DNoHelmet",
    "DNoHelmetP0NoHelmet",
    "DNoHelmetP0NoHelmetP1NoHelmet",
    "DNoHelmetP0NoHelmetP1NoHelmetP2NoHelmet",
    "DNoHelmetP1Helmet",
    "DNoHelmetP1NoHelmet",
    "DNoHelmetP1NoHelmetP2NoHelmet",
]

NUM_OBJECT_CLASSES = len(CLASS_NAMES)
NUM_CLASSES = NUM_OBJECT_CLASSES + 1

LABEL_TO_ID = {
    name: index + 1
    for index, name in enumerate(CLASS_NAMES)
}

ID_TO_LABEL = {
    index: name
    for name, index in LABEL_TO_ID.items()
}


# ============================================================
# DATASET
# ============================================================

class HelmetDataset(Dataset):

    def __init__(self, split="train"):

        if split not in ("train", "val", "test"):
            raise ValueError(
                "split must be 'train', 'val', or 'test'"
            )

        self.split = split

        for path in (IMAGE_ROOT, ANNOTATION_ROOT, SPLIT_CSV):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Dataset path not found: {path}"
                )

        split_df = pd.read_csv(SPLIT_CSV)

        split_names = {
            "training": "train",
            "validation": "val",
            "test": "test",
        }

        split_df["split"] = (
            split_df["Set"]
            .astype(str)
            .str.lower()
            .map(split_names)
        )

        if split_df["split"].isna().any():
            raise ValueError("Unexpected split name in data_split.csv")

        video_to_split = dict(
            zip(split_df["video_id"], split_df["split"])
        )

        part1_videos = sorted(
            name
            for name in os.listdir(IMAGE_ROOT)
            if os.path.isdir(os.path.join(IMAGE_ROOT, name))
        )

        self.annotations = defaultdict(list)

        self.annotation_files_loaded = 0
        self.unrecognized_labels = 0
        self.invalid_boxes = 0

        for video_id in part1_videos:

            csv_path = os.path.join(
                ANNOTATION_ROOT, f"{video_id}.csv"
            )

            if not os.path.isfile(csv_path):
                raise FileNotFoundError(
                    f"Missing annotation CSV: {csv_path}"
                )

            df = pd.read_csv(csv_path)
            self.annotation_files_loaded += 1

            required = {
                "frame_id", "x", "y", "w", "h", "label"
            }

            missing = required - set(df.columns)

            if missing:
                raise ValueError(
                    f"{csv_path} is missing columns: {missing}"
                )

            for row in df.itertuples(index=False):

                original_label = str(row.label).strip()

                # Preserve every original annotation class.
                class_id = LABEL_TO_ID.get(original_label)

                if class_id is None:
                    self.unrecognized_labels += 1
                    continue

                try:
                    frame_id = int(row.frame_id)
                    x = float(row.x)
                    y = float(row.y)
                    w = float(row.w)
                    h = float(row.h)
                except (ValueError, TypeError):
                    self.invalid_boxes += 1
                    continue

                if (
                    frame_id < 1
                    or frame_id > 100
                    or w <= 0
                    or h <= 0
                ):
                    self.invalid_boxes += 1
                    continue

                self.annotations[
                    (video_id, frame_id)
                ].append({
                    "box": [x, y, x + w, y + h],
                    "label": class_id,
                })

        self.samples = []

        for video_id in part1_videos:

            if video_to_split.get(video_id) != split:
                continue

            video_dir = os.path.join(IMAGE_ROOT, video_id)

            for frame_id in range(1, 101):

                image_path = os.path.join(
                    video_dir, f"{frame_id:02d}.jpg"
                )

                if os.path.isfile(image_path):
                    self.samples.append({
                        "image_path": image_path,
                        "video_id": video_id,
                        "frame_id": frame_id,
                    })

        if not self.samples:
            raise RuntimeError(
                f"No images found for split '{split}'"
            )

        print(f"\nLoaded {split} dataset")
        print("Annotation CSVs:", self.annotation_files_loaded)
        print("Images:", len(self.samples))
        print("Unrecognized labels:", self.unrecognized_labels)
        print("Invalid annotation rows:", self.invalid_boxes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = cv2.imread(sample["image_path"])

        if image is None:
            raise RuntimeError(
                f"Could not read image: {sample['image_path']}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = torch.from_numpy(image.copy())
        image = image.permute(2, 0, 1).float() / 255.0

        height, width = image.shape[1], image.shape[2]

        boxes = []
        labels = []

        for item in self.annotations.get(
            (sample["video_id"], sample["frame_id"]), []
        ):

            xmin, ymin, xmax, ymax = item["box"]

            xmin = max(0.0, min(xmin, float(width)))
            xmax = max(0.0, min(xmax, float(width)))
            ymin = max(0.0, min(ymin, float(height)))
            ymax = max(0.0, min(ymax, float(height)))

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(item["label"])

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (
                (boxes[:, 2] - boxes[:, 0])
                * (boxes[:, 3] - boxes[:, 1])
            ),
            "iscrowd": torch.zeros(
                (len(labels),), dtype=torch.int64
            ),
        }

        return image, target
