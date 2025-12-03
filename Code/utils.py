import numpy as np
import torch

def radial_average(psd):
    y, x = np.indices(psd.shape)
    center = np.array(psd.shape) / 2.0
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)
    return radial_profile


def generate_images(I_smooth, I_sharp, k_smooth, k_sharp, epsilon=1e-6):
    """
    I_smooth, I_sharp: [B,1,H,W] images
    k_smooth, k_sharp: [B,1] scalars from generator
    Returns single [1,1,H,W] images for discriminator
    """
    # FFT
    F_smooth = torch.fft.fft2(I_smooth)
    F_sharp = torch.fft.fft2(I_sharp)

    # Expand scalars to broadcast
    ks = k_smooth.view(-1, 1, 1, 1).clamp(min=1e-6)
    kh = k_sharp.view(-1, 1, 1, 1).clamp(min=1e-6)

    OTF_s2h = kh / (ks + epsilon)
    OTF_h2s = ks / (kh + epsilon)

    # Apply frequency scaling
    I_gen_sharp = torch.real(torch.fft.ifft2(F_smooth * OTF_s2h))
    I_gen_smooth = torch.real(torch.fft.ifft2(F_sharp * OTF_h2s))

    # Average over batch dimension to get single image
    I_gen_sharp = I_gen_sharp.mean(dim=0, keepdim=True)  # [1,1,H,W]
    I_gen_smooth = I_gen_smooth.mean(dim=0, keepdim=True)  # [1,1,H,W]

    return I_gen_sharp, I_gen_smooth

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

