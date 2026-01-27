import torch
import torch.nn as nn

class ResBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # [B,1,512,512] -> [B,16,256,256]
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock2D(16),
            
            # [B,16,256,256] -> [B,32,128,128]
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock2D(32),
            
            # [B,32,128,128] -> [B,64,64,64]
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock2D(64),
            
            # [B,64,64,64] -> [B,128,32,32]
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock2D(128),
            
            # [B,128,32,32] -> [B,256,16,16]
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            ResBlock2D(256),
            
            # [B,256,16,16] -> [B,1,1,1]
            nn.Conv2d(256, 1, 4, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Accept [B,H,W] or [B,1,H,W]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x