"""Pascal VOC -> YOLOv1 dataset.
 
Target layout for each grid cell, shape (S, S, 5*B + C):
 
    [x, y, w, h, conf] * B  |  one-hot class (C)
 
x, y : box centre as an offset inside its cell, in [0, 1)
w, h : box size relative to the whole image, in [0, 1]
"""

from utils import parse_xml
from config import DATA_DIR
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

        # Lazy loading: only read the text file of IDs here (takes 0.005s)
        self.ids = (self.root / "ImageSets" / "Main" / f"{split}.txt").read_text().split()

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

    def _random_scale_sample(self, img, labels, scale_range=(0.8, 1.2)):
        """
        Randomly scales the image and adjusts bounding box coordinates.
        - scale <= 1.0: Zoom-out with random offset placement on a neutral padded canvas.
        - scale > 1.0:  Zoom-in / random crop patch sampling with box filtering.
        """
        scale = random.uniform(scale_range[0], scale_range[1])
        W, H = img.size

        if scale <= 1.0:
            # Zoom-out / random padding placement
            new_w, new_h = max(int(W * scale), 1), max(int(H * scale), 1)
            resized_img = img.resize((new_w, new_h), Image.BILINEAR)

            dx = random.randint(0, W - new_w)
            dy = random.randint(0, H - new_h)

            canvas = Image.new("RGB", (W, H), (124, 116, 104))
            canvas.paste(resized_img, (dx, dy))

            new_labels = labels.clone()
            new_labels[:, 1] = labels[:, 1] * scale + (dx / W)
            new_labels[:, 2] = labels[:, 2] * scale + (dy / H)
            new_labels[:, 3] = labels[:, 3] * scale
            new_labels[:, 4] = labels[:, 4] * scale
            return canvas, new_labels
        else:
            # Zoom-in / random crop patch sampling
            new_w, new_h = int(W * scale), int(H * scale)
            resized_img = img.resize((new_w, new_h), Image.BILINEAR)

            dx = random.randint(0, new_w - W)
            dy = random.randint(0, new_h - H)

            cropped_img = resized_img.crop((dx, dy, dx + W, dy + H))

            cx_new = labels[:, 1] * scale - (dx / W)
            cy_new = labels[:, 2] * scale - (dy / H)
            w_new = labels[:, 3] * scale
            h_new = labels[:, 4] * scale

            # Keep boxes whose centers remain within the crop
            valid_mask = (cx_new > 0.02) & (cx_new < 0.98) & (cy_new > 0.02) & (cy_new < 0.98)
            if valid_mask.sum() > 0:
                new_labels = torch.stack([
                    labels[valid_mask, 0],
                    cx_new[valid_mask].clamp(0.01, 0.99),
                    cy_new[valid_mask].clamp(0.01, 0.99),
                    w_new[valid_mask].clamp(0.01, 0.99),
                    h_new[valid_mask].clamp(0.01, 0.99)
                ], dim=1)
                return cropped_img, new_labels
            else:
                return img, labels

    def _augment(self, img, labels):
        # 1. Random Scaling & Patch Sampling (50% probability)
        if random.random() < 0.5 and len(labels) > 0:
            img, labels = self._random_scale_sample(img, labels, scale_range=(0.8, 1.2))

        # 2. Random Horizontal Flip (50% probability)
        if random.random() < 0.5 and len(labels) > 0:
            img = TF.hflip(img)
            labels[:, 1] = 1.0 - labels[:, 1] # if img is flipped then boxes are flipped also

        # 3. Color Jitter
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
        img_id = self.ids[idx]
        img = Image.open(self.root / "JPEGImages" / f"{img_id}.jpg").convert("RGB")
        
        # Load label on-demand; multi-worker DataLoader handles this in parallel across CPU cores
        labels = self._load_labels(img_id)

        img = self.tf(img)
        if self.augment:
            img, labels = self._augment(img, labels)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 
        return img, self.encode(labels)


if __name__ == "__main__":
    from config import DATA_DIR
 
    ds = VOCYolo(DATA_DIR, split="trainval", augment=True)
    img, target = ds[0]
    print(f"samples: {len(ds)}")
    print(f"image:   {tuple(img.shape)}")
    print(f"target:  {tuple(target.shape)}")
    print(f"objects encoded: {int(target[..., 4].sum())}  (raw labels: {len(ds._load_labels(ds.ids[0]))})")
