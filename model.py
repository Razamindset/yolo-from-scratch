import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.head = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=1024, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),                       # 14 -> 7
            nn.Conv2d(kernel_size=1, in_channels=1024, out_channels=30)
        )


    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return torch.sigmoid(x.permute(0, 2, 3, 1))

model = YOLOv1()
print(model(torch.rand(2, 3, 448, 448)).shape)   # expect (2, 7, 7, 30)