import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ROOT
from dataset import VOCYolo
from model import YOLOv1
from loss import YoloLoss

EPOCHS = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

criterion = YoloLoss()

def main():
    train_ds = VOCYolo(ROOT, split="trainval", augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
    )

    model = YOLOv1().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    imgs, targets = next(iter(train_loader))
    imgs, targets = imgs.to(device), targets.to(device)

    for step in range(100):
        pred = model(imgs)
        loss = criterion(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(step, loss.item())

if __name__ == "__main__":
    main()
