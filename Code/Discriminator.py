import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),   # [B,1,512,512] to [B,16,256,256]
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 4, stride=2, padding=1),  # [B,16,256,256] to [B,32,128,128]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [B,32,128,128] to [B,64,64,64]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 1, 4, padding=0),             # [B,64,64,64] to [B,1,61,61]
            nn.AdaptiveAvgPool2d((1, 1)),               # [B,1,1,1]
            nn.Sigmoid()                                # ensure output in [0,1]
        )

    def forward(self, x):
        # Accept [B,H,W] or [B,1,H,W]
        if x.ndim == 3:           # add channel
            x = x.unsqueeze(1)
        x = self.model(x)         # [B,1,1,1]
        x = x.view(x.size(0), -1) # [B,1] → one scalar per image
        return x
