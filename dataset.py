from utils import parse_xml
from config import ROOT
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import torch

class_counts = Counter()
size_counts = Counter()
objs_per_image = []
difficult_total = 0

with open(f"{ROOT}/ImageSets/Main/trainval.txt") as f:
    ids = f.read().split()

for i in tqdm(ids):
    w, h, boxes = parse_xml(f"{ROOT}/Annotations/{i}.xml")
    size_counts[(w, h)] += 1
    objs_per_image.append(len(boxes))
    for name, xmin, ymin, xmax, ymax, difficult in boxes:
        class_counts[name] += 1
        difficult_total += difficult


CLASSES = sorted(class_counts.keys())
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

def convert_box(xmin, ymin, xmax, ymax, w, h):
    cx = (xmin + xmax) / (2*w)
    cy = (ymin + ymax) / (2*h)

    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h

    return cx, cy, bw, bh 

def locate_cell(cx, cy, S=7):
    row = int(cy * 7)
    col = int(cx * 7)

    x_off = cx * S - col
    y_off = cy * S - row
    
    return (row, col, x_off, y_off)
    


def build_target(boxes, w, h, S=7):
    target = torch.zeros(S, S, 30)

    for name, xmin, ymin, xmax, ymax, difficult in boxes:

        cx, cy, bw, bh = convert_box(xmin, ymin, xmax, ymax, w, h)

        row, col, x_off, y_off = locate_cell(cx, cy, S)

        if target[row, col, 4] == 1:
            continue

        target[row, col, 0:5] = torch.tensor((x_off, y_off, bw, bh, 1))

        target[row, col, 5:10] = torch.tensor((x_off, y_off, bw, bh, 1))

        class_idx = class_to_idx[name]

        target[row, col, 10 + class_idx] = 1

    return target

class VOCYolo(Dataset):
    def __init__(self, root, ids):
        self.root = root
        self.ids = ids
        self.tf = T.Compose([T.Resize((448, 448)), T.ToTensor()])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        w, h, boxes = parse_xml(
            f"{self.root}/Annotations/{i}.xml"
        )         
        target = build_target(boxes, w, h)                     # 2. build the 7x7x30 target
        img = Image.open(f"{self.root}/JPEGImages/{i}.jpg").convert("RGB")   # 3. open the JPEG
        img = self.tf(img)                # resize + make tensor
        return img, target