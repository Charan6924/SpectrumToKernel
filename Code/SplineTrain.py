import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PSDDataset import PSDDataset
from SplineEstimator import KernelEstimator, FixedSplineLayer
from Discriminator import Discriminator
from utils import generate_images, normalize
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

# Setup and Hyperparameters
save_dir = "generated_images_spline"
kernel_dir = "spline_kernels"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(kernel_dir, exist_ok=True) 
device = "cuda" if torch.cuda.is_available() else "cpu"
alpha = 0.1
net = KernelEstimator().to(device)
dataset = PSDDataset(root_dir=r"D:\Charan work file\KernelEstimator\Data_Root")
loader = DataLoader(dataset, batch_size=1, shuffle=True)
dataset_phantom = PSDDataset(root_dir=r"D:\Charan work file\KernelEstimator\Phantom_Root", mode='phantom')
loader_phantom = DataLoader(dataset_phantom, batch_size=1, shuffle=True)
phantom_iter = iter(loader_phantom)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
bce = nn.BCELoss().to(device)
l1 = nn.L1Loss().to(device)


# Loop start
for epoch in range(40):
    net.train()
    running_total_loss = 0.0
    running_recon_loss = 0.0
    running_phantom_loss = 0.0

    for idx, (I_smooth, I_sharp, psd_smooth, psd_sharp) in enumerate(loader):
        I_smooth = I_smooth.to(device)
        I_sharp = I_sharp.to(device)
        try:
            psd_phantom, mtf_phantom_gt = next(phantom_iter)
        except StopIteration:
            phantom_iter = iter(loader_phantom)
            psd_phantom, mtf_phantom_gt = next(phantom_iter)
            
        psd_phantom = psd_phantom.to(device)
        mtf_phantom_pred = net(psd_phantom)
        mtf_phantom_gt = mtf_phantom_gt.to(device)
        psd_smooth = psd_smooth.to(device)
        psd_sharp = psd_sharp.to(device)

        mtf_smooth = net(psd_smooth) 
        mtf_sharp  = net(psd_sharp)
        I_gen_sharp, I_gen_smooth = generate_images(I_smooth, I_sharp, mtf_smooth, mtf_sharp)

        #saving the images
        if idx == 0:
            with torch.no_grad():
                n_samples = min(4, I_sharp.size(0))
                for i in range(n_samples):
                    imgs = torch.stack([
                        normalize(I_smooth[i]),
                        normalize(I_gen_smooth[i]),
                        normalize(I_sharp[i]),
                        normalize(I_gen_sharp[i]),
                    ])
                    save_path = os.path.join(save_dir, f"epoch{epoch + 1}_sample{i + 1}.png")
                    save_image(imgs, save_path, nrow=2)

                curve_smooth_np = mtf_smooth[0, 0].detach().cpu().numpy() 
                curve_sharp_np  = mtf_sharp[0, 0].detach().cpu().numpy()
                curve_ph_pred_np = mtf_phantom_pred[0, 0].detach().cpu().numpy() 
                curve_ph_gt_np = mtf_phantom_gt[0, 0].detach().cpu().numpy()
                num_points = len(curve_smooth_np)
                freq_axis = np.linspace(0, 0.5, num_points)

                # plotting
                plt.figure(figsize=(8, 6))
                plt.plot(freq_axis, curve_smooth_np, label='Predicted Smooth MTF', color='blue', linewidth=2)
                plt.plot(freq_axis, curve_sharp_np, label='Predicted Sharp MTF', color='red', linewidth=2)
                plt.plot(freq_axis, curve_ph_pred_np, label='Predicted Phantom MTF', color='green', linestyle='--', linewidth=2)
                plt.plot(freq_axis, curve_ph_gt_np, label='Ground Truth Phantom MTF', color='orange', linestyle='--', linewidth=2)
                plt.title(f'B-Spline at - Epoch {epoch + 1}')
                plt.xlim(0, 0.5)
                plt.ylim(0, 1.05) # MTF shouldn't go much above 1.0
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.savefig(os.path.join(kernel_dir, f'spline_epoch{epoch + 1}.png'))
                plt.close()

        # calculating loss and updating parameters
        recon_loss = l1(I_gen_sharp, I_sharp) + l1(I_gen_smooth, I_smooth)
        recon_loss = recon_loss / 2.0 
        loss_phantom = l1(mtf_phantom_pred, mtf_phantom_gt)

        total_loss = (recon_loss * alpha) + ((1-alpha) *loss_phantom)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        running_total_loss += total_loss.item()
        running_recon_loss += recon_loss.item()
        running_phantom_loss += loss_phantom.item()

    avg_total = running_total_loss / len(loader)
    avg_recon = running_recon_loss / len(loader)
    avg_ph    = running_phantom_loss / len(loader)

    print(f"Epoch [{epoch+1}/40]  Total Loss: {avg_total:.4f} | Recon: {avg_recon:.4f} | Phantom: {avg_ph:.4f}")
torch.save(net, "kernel_estimator_full.pth")