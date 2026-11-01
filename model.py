import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2),
        )
        
        self.flatten = nn.Flatten()
        
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.ff = nn.Linear(64 * 224 * 224, 1470)
    
    def forward(self, x):
        x = self.block(x)
        x = self.flatten(x)
        x = self.pool(x)
        x = self.ff(x)
    
        return x.reshape(-1, 7, 7, 30)


model = TinyModel(in_channels=3, out_channels=64)

dummy_input = torch.randn(1, 3, 448, 448)

predictions = model(dummy_input)

print(predictions.shape)