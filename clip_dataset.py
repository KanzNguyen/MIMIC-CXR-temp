import os
import json
import pandas as pd
from typing import Optional, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

try:
    import open_clip
except ImportError:  # tokenizer fallback (raise later if used)
    open_clip = None

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_clip_transform(train: bool, image_size: int = 224) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize(256),
            T.RandomCrop(image_size),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.0),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    else:
        return T.Compose([
            T.Resize(224),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])


class MimicCxrClipDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        base_path: str,
        split: str,
        text_ctx_len: int = 256,
        is_train: bool = True,
    ) -> None:
        if open_clip is None:
            raise ImportError("open_clip_torch is required. Please install open-clip-torch.")

        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"].str.lower() == split.lower()].reset_index(drop=True)
        self.base_path = base_path
        self.transform = build_clip_transform(train=is_train)
        self.text_ctx_len = text_ctx_len

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: Optional[str]) -> Optional[Image.Image]:
        if isinstance(rel_path, str) and len(rel_path) > 0 and rel_path != 'nan':
            img_path = rel_path
            if not os.path.isabs(img_path):
                img_path = os.path.join(self.base_path, rel_path)
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                return img
        return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        frontal_rel = row.get('image_frontal_path', None)
        lateral_rel = row.get('image_lateral_path', None)
        findings = str(row.get('findings', ""))
        label_vector_raw = row.get('label_vector', None)

        img_frontal = self._load_image(frontal_rel)
        img_lateral = self._load_image(lateral_rel)

        if img_frontal is None and img_lateral is None:
            raise FileNotFoundError(f"No valid images at index {idx}")

        if img_frontal is not None:
            img_frontal = self.transform(img_frontal)
        else:
            img_frontal = torch.zeros(3, 224, 224)

        if img_lateral is not None:
            img_lateral = self.transform(img_lateral)
        else:
            img_lateral = torch.zeros(3, 224, 224)

        # Clean text: lowercase and basic cleanup
        text_clean = findings.strip().lower()
        text_tokens = open_clip.tokenize([text_clean], context_length=self.text_ctx_len)[0]

        # Parse label vector
        labels = None
        if isinstance(label_vector_raw, str) and len(label_vector_raw) > 0:
            try:
                labels = torch.tensor(json.loads(label_vector_raw), dtype=torch.float32)
            except Exception:
                try:
                    labels = torch.tensor(eval(label_vector_raw), dtype=torch.float32)
                except Exception:
                    labels = None

        sample = {
            'image_frontal': img_frontal,
            'image_lateral': img_lateral,
            'text_tokens': text_tokens,
            'labels': labels,
            'index': idx,
        }
        return sample


def create_clip_dataloaders(
    csv_path: str,
    base_path: str,
    batch_size: int = 128,
    num_workers: int = 4,
    text_ctx_len: int = 256,
):
    train_set = MimicCxrClipDataset(
        csv_path=csv_path,
        base_path=base_path,
        split='train',
        text_ctx_len=text_ctx_len,
        is_train=True,
    )
    val_set = MimicCxrClipDataset(
        csv_path=csv_path,
        base_path=base_path,
        split='val',
        text_ctx_len=text_ctx_len,
        is_train=False,
    )
    test_set = MimicCxrClipDataset(
        csv_path=csv_path,
        base_path=base_path,
        split='test',
        text_ctx_len=text_ctx_len,
        is_train=False,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Infer number of classes from first non-empty label
    num_classes = None
    for ds in [train_set, val_set, test_set]:
        for i in range(len(ds)):
            item = ds[i]
            if item['labels'] is not None:
                num_classes = item['labels'].numel()
                break
        if num_classes is not None:
            break

    return train_loader, val_loader, test_loader, num_classes
