"""Pascal VOC -> YOLOv1 dataset.
 
Target layout for each grid cell, shape (S, S, 5*B + C):
 
    [x, y, w, h, conf] * B  |  one-hot class (C)
 
x, y : box centre as an offset inside its cell, in [0, 1)
w, h : box size relative to the whole image, in [0, 1]
"""

from utils import parse_xml
from config import ROOT
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import torch
import random

# Alphabetical order (same as sorted() over the class names found in VOC).
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}


class VOCYolo(Dataset):
    def __init__(self, root, split="trainval", S=7, B=2, img_size=448, augment=False, keep_difficult=False):
        self.root = Path(root)
        self.S, self.B, self.C = S, B, len(VOC_CLASSES)
        self.augment = augment
        self.keep_difficult = keep_difficult
        
        self.tf = T.Resize((img_size, img_size))
        self.jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)

        self.ids = (self.root / "ImageSets" / "Main" / f"{split}.txt").read_text().split()
        self.labels = [self._load_labels(i) for i in self.ids]

    def _load_labels(self, img_id):
        """Returns a (N, 5) tensor of [class_idx, cx, cy, w, h], all normalised to the image."""
        w, h, boxes = parse_xml(self.root / "Annotations" / f"{img_id}.xml")
 
        rows = []
        for name, xmin, ymin, xmax, ymax, difficult in boxes:
            if difficult and not self.keep_difficult:
                continue
            rows.append([
                CLASS_TO_IDX[name],
                (xmin + xmax) / (2 * w),
                (ymin + ymax) / (2 * h),
                (xmax - xmin) / w,
                (ymax - ymin) / h,
            ])
        return torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)

    def _augment(self, img, labels):
        if random.random() < 0.5:
            img = TF.hflip(img)
            labels[:, 1] = 1.0 - labels[:, 1] # if img is flipped then boxes are flipped also

        img = self.jitter(img)

        return img, labels

    def encode(self, labels):
        """(N, 5) normalised boxes -> (S, S, 5*B + C) target tensor."""
        S, B = self.S, self.B
        target = torch.zeros(S, S, 5*B + self.C)

        for cls, cx, cy, w, h in labels.tolist():
            # min() guards the edge case cx == 1.0, which would index cell S.
            col = min(int(cx * S), S - 1)
            row = min(int(cy * S), S - 1)

            # One object per cell: if the cell is taken, the extra object is dropped (YOLOv1 limitation).
            if target[row, col, 4] == 1:
                continue

            box = torch.tensor([cx * S - col, cy * S - row, w, h, 1.0])

            for b in range(B):
                target[row, col, 5* b: 5 * b + 5] = box

            target[row, col, 5 * B + int(cls)] = 1.0

        return target

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = Image.open(self.root / "JPEGImages" / f"{self.ids[idx]}.jpg").convert("RGB")
        labels = self.labels[idx].clone()  # clone: augmentation edits in place
 
        img = self.resize(img)
        if self.augment:
            img, labels = self._augment(img, labels)
 
        return TF.to_tensor(img), self.encode(labels)


if __name__ == "__main__":
    from config import ROOT
 
    ds = VOCYolo(ROOT, split="trainval", augment=True)
    img, target = ds[0]
    print(f"samples: {len(ds)}")
    print(f"image:   {tuple(img.shape)}")
    print(f"target:  {tuple(target.shape)}")
    print(f"objects encoded: {int(target[..., 4].sum())}  (raw labels: {len(ds.labels[0])})")
