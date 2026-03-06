"""
dataset.py - PyTorch Dataset for License Plate OCR using TrOCR
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor


class LicensePlateDataset(Dataset):
    """
    PyTorch Dataset for license plate OCR.

    Loads images and their corresponding plate text labels,
    preprocesses via TrOCRProcessor (pixel_values) and tokenizes
    the target text into label IDs for the decoder.
    """

    def __init__(
        self,
        samples: List[Tuple[str, str]],
        images_dir: str,
        processor: TrOCRProcessor,
        max_target_length: int = 20,
    ):
        self.samples = samples
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename, plate_text = self.samples[idx]
        image_path = self.images_dir / filename

        # TrOCRProcessor expects PIL RGB; handles ViT resize + normalize internally
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)
        # Shape: [3, 384, 384]

        # Tokenize label text, pad to fixed length
        labels = self.processor.tokenizer(
            plate_text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        # Shape: [max_target_length]

        # Replace pad_token_id with -100 so CrossEntropyLoss ignores padding
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,   # float32 [3, 384, 384]
            "labels": labels,               # int64  [max_target_length]
        }


def load_annotations(annotations_file: str) -> List[Tuple[str, str]]:
    """
    Load JSON annotations file.
    Expected format: {"img001.jpg": "372OHA02", ...}
    Returns list of (filename, plate_text) tuples.
    """
    with open(annotations_file, "r", encoding="utf-8") as f:
        annotations: Dict[str, str] = json.load(f)

    samples = []
    for filename, plate_text in annotations.items():
        cleaned = plate_text.strip().upper()
        if cleaned:
            samples.append((filename, cleaned))

    return samples


def create_splits(
    samples: List[Tuple[str, str]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    random_seed: int = 42,
) -> Tuple[List, List, List]:
    """Deterministic 80/10/10 split."""
    random.seed(random_seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train_s = shuffled[:train_end]
    val_s   = shuffled[train_end:val_end]
    test_s  = shuffled[val_end:]

    print(f"[Dataset] Total: {n} | Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")
    return train_s, val_s, test_s


def build_datasets(
    annotations_file: str,
    images_dir: str,
    processor: TrOCRProcessor,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    random_seed: int = 42,
    max_target_length: int = 20,
) -> Tuple["LicensePlateDataset", "LicensePlateDataset", "LicensePlateDataset"]:
    """Load annotations, split, and build all three Dataset objects."""
    all_samples = load_annotations(annotations_file)
    train_s, val_s, test_s = create_splits(all_samples, train_ratio, val_ratio, random_seed)

    train_ds = LicensePlateDataset(train_s, images_dir, processor, max_target_length)
    val_ds   = LicensePlateDataset(val_s,   images_dir, processor, max_target_length)
    test_ds  = LicensePlateDataset(test_s,  images_dir, processor, max_target_length)

    return train_ds, val_ds, test_ds
