import torch
import torch.nn as nn
from Discriminator import Discriminator  # make sure this is the corrected class

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate discriminator
D_sharp = Discriminator().to(device)

# Create a dummy 2D image batch [B,H,W] or [B,1,H,W]
B, H, W = 1, 512, 512
I_sharp = torch.randn(B, H, W, device=device)  # single 2D slice
I_smooth = torch.randn(B, H, W, device=device)

# Forward pass
out_real_sharp = D_sharp(I_sharp)
out_real_smooth = D_sharp(I_smooth)

# Check shapes
print("I_sharp shape:", I_sharp.shape)
print("out_real_sharp shape:", out_real_sharp.shape)
print("I_smooth shape:", I_smooth.shape)
print("out_real_smooth shape:", out_real_smooth.shape)

# Create matching BCE labels
real_label = torch.ones(B, 1, device=device)
fake_label = torch.zeros(B, 1, device=device)

# Test BCE loss
bce = nn.BCELoss()
loss_sharp = bce(out_real_sharp, real_label)
loss_smooth = bce(out_real_smooth, real_label)

print("BCE loss sharp:", loss_sharp.item())
print("BCE loss smooth:", loss_smooth.item())
