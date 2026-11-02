import os
import csv
import time
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.helmet_dataset import (
    HelmetDataset,
    CLASS_NAMES,
    LABEL_TO_ID,
    NUM_CLASSES,
)
from training.model import build_model


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = "/mnt/ssd/helmet_project"
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0005
STEP_SIZE = 3
GAMMA = 0.1
NUM_WORKERS = 0

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LATEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR, "frcnn_16class_latest.pth"
)

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR, "frcnn_16class_best.pth"
)

HISTORY_CSV = os.path.join(
    LOG_DIR, "training_history.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA LOADER
# ============================================================

def collate_fn(batch):
    return tuple(zip(*batch))


def create_loader(split):
    dataset = HelmetDataset(split)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================

def atomic_save(checkpoint, path):
    temporary_path = path + ".tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    history,
    best_val_loss,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
        "class_names": CLASS_NAMES,
        "label_to_id": LABEL_TO_ID,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "step_size": STEP_SIZE,
        "gamma": GAMMA,
    }

    atomic_save(checkpoint, path)

    size_mb = os.path.getsize(path) / (1024 ** 2)

    print(f"Checkpoint saved: {path}")
    print(f"Checkpoint size: {size_mb:.1f} MB")


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("num_classes") != NUM_CLASSES:
        raise RuntimeError(
            "Checkpoint class count does not match current model."
        )

    if checkpoint.get("class_names") != CLASS_NAMES:
        raise RuntimeError(
            "Checkpoint class mapping does not match current dataset."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return (
        checkpoint["epoch"] + 1,
        checkpoint["history"],
        checkpoint["best_val_loss"],
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()

    running_loss = 0.0
    batches = len(loader)

    start_time = time.time()

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]

        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)

        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}, "
                f"batch {batch_index}: {total_loss.item()}"
            )

        total_loss.backward()

        # Protect against extreme gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=10.0
        )

        optimizer.step()

        loss_value = total_loss.item()
        running_loss += loss_value

        if batch_index == 1 or batch_index % 50 == 0 or batch_index == batches:
            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_index}/{batches} | "
                f"Loss {loss_value:.4f} | "
                f"Average {running_loss / batch_index:.4f} | "
                f"Elapsed {elapsed / 60:.1f} min",
                flush=True,
            )

    return running_loss / max(1, batches)


# ============================================================
# VALIDATION LOSS
# ============================================================

@torch.no_grad()
def validate_loss(model, loader, device):
    # Torchvision detection models return losses in train mode.
    # Keep BatchNorm layers in evaluation mode to avoid updating
    # their running statistics during validation.
    model.train()

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    running_loss = 0.0
    batches = len(loader)

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]

        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite validation loss at batch {batch_index}"
            )

        running_loss += total_loss.item()

    return running_loss / max(1, batches)


# ============================================================
# HISTORY
# ============================================================

def save_history(history):
    if not history:
        return

    with open(HISTORY_CSV, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "learning_rate",
                "epoch_minutes",
            ],
        )

        writer.writeheader()
        writer.writerows(history)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Total target epoch count.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("16-CLASS FASTER R-CNN TRAINING")
    print("=" * 65)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Classes including background:", NUM_CLASSES)
    print("Batch size:", BATCH_SIZE)
    print("Target epochs:", args.epochs)
    print("Checkpoint directory:", CHECKPOINT_DIR)

    train_loader = create_loader("train")
    val_loader = create_loader("val")

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    model = build_model(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=GAMMA,
    )

    start_epoch = 1
    history = []
    best_val_loss = float("inf")

    if args.resume:
        if not os.path.isfile(LATEST_CHECKPOINT):
            raise FileNotFoundError(
                "Cannot resume: latest checkpoint does not exist."
            )

        start_epoch, history, best_val_loss = load_checkpoint(
            LATEST_CHECKPOINT,
            model,
            optimizer,
            scheduler,
            device,
        )

        print("Resuming at epoch:", start_epoch)

    if start_epoch > args.epochs:
        print("Checkpoint already reached the requested epoch count.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        print("\n" + "=" * 65)
        print(f"STARTING EPOCH {epoch}/{args.epochs}")
        print("=" * 65)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
        )

        val_loss = validate_loss(
            model,
            val_loader,
            device,
        )

        scheduler.step()

        epoch_minutes = (time.time() - epoch_start) / 60.0

        print(f"\nEpoch {epoch} summary")
        print(f"Training loss: {train_loss:.6f}")
        print(f"Validation loss: {val_loss:.6f}")
        print(f"Epoch duration: {epoch_minutes:.1f} minutes")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_minutes": epoch_minutes,
        })

        save_history(history)

        save_checkpoint(
            LATEST_CHECKPOINT,
            epoch,
            model,
            optimizer,
            scheduler,
            history,
            best_val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                BEST_CHECKPOINT,
                epoch,
                model,
                optimizer,
                scheduler,
                history,
                best_val_loss,
            )

        print("Latest checkpoint updated.")
        print("Best validation loss:", best_val_loss)

    print("\nTraining completed.")
    print("Latest checkpoint:", LATEST_CHECKPOINT)
    print("Best checkpoint:", BEST_CHECKPOINT)
    print("Training history:", HISTORY_CSV)


if __name__ == "__main__":
    main()
