import torch
import torch.nn as nn
import numpy as np
import nibabel
import matplotlib.pyplot as plt
from PSDDataset import PSDDataset
from KernelEstimator import KernelEstimator
from Discriminator import Discriminator
from utils import generate_images, radial_average, normalize
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import csv

save_dir = "generated_images"
kernel_dir = "kernels"
os.makedirs(save_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
net = KernelEstimator().to(device)
dataset = PSDDataset(root_dir = r"D:\Charan work file\KernelEstimator\Data_Root")
loader = DataLoader(dataset,batch_size = 1, shuffle = True)
D_sharp = Discriminator().to(device)
D_smooth = Discriminator().to(device)
bce = nn.BCELoss().to(device)
l1 = nn.L1Loss().to(device)

opt_G = torch.optim.Adam(net.parameters(), lr=1e-4)
opt_Ds = torch.optim.Adam(
    list(D_sharp.parameters()) + list(D_smooth.parameters()), lr=1e-4
)

for epoch in range(40):
    net.train()
    D_sharp.train()
    D_smooth.train()

    total_G_loss, total_D_loss = 0.0, 0.0
    

    for idx, (I_smooth, I_sharp, psd_smooth, psd_sharp) in enumerate(loader):
        I_smooth = I_smooth.to(device)
        I_sharp = I_sharp.to(device)
        psd_smooth = psd_smooth.to(device)
        psd_sharp = psd_sharp.to(device)

        # ====== Generator forward ======
        k_smooth = net(psd_smooth)
        k_sharp  = net(psd_sharp)
        I_gen_sharp, I_gen_smooth = generate_images(I_smooth, I_sharp, k_smooth, k_sharp)

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
                k_smooth_np = k_smooth[0, 0].cpu().numpy()  # turn into np array
                k_sharp_np = k_sharp[0, 0].cpu().numpy()
                mtf_smooth = np.abs(np.fft.rfft(k_smooth_np))
                mtf_sharp = np.abs(np.fft.rfft(k_sharp_np))
                if mtf_smooth[0] > 1e-6:
                    mtf_smooth = mtf_smooth / mtf_smooth[0]
                if mtf_sharp[0] > 1e-6:
                    mtf_sharp = mtf_sharp / mtf_sharp[0]
                N = len(k_smooth_np)  #N = 363
                freq_axis = np.fft.rfftfreq(N, d=1.0)
                plt.figure()
                plt.plot(freq_axis * N, mtf_smooth, label='Smooth MTF')
                plt.plot(freq_axis * N, mtf_sharp, label='Sharp MTF')
                plt.title(f'MTF (Modulation Transfer Function) at Epoch {epoch + 1}')
                plt.xlabel('Frequency')
                plt.ylabel('MTF')
                plt.xlim(0,250)
                plt.ylim(0, 1.1)
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(kernel_dir, f'mtf_epoch{epoch + 1}.png'))
                plt.close()


        # ====== --- Train Discriminators --- ======
        opt_Ds.zero_grad()

        # Real labels = 1, Fake = 0
        B = I_gen_sharp.size(0)  # 363
        real_label = torch.ones(B, 1, device=device)
        fake_label = torch.zeros(B, 1, device=device)

        # D_sharp
        out_real_sharp = D_sharp(I_sharp)
        out_fake_sharp = D_sharp(I_gen_sharp.detach())
        loss_D_sharp = 0.5 * (bce(out_real_sharp, real_label) + bce(out_fake_sharp, fake_label))

        # D_smooth
        out_real_smooth = D_smooth(I_smooth)
        out_fake_smooth = D_smooth(I_gen_smooth.detach())
        loss_D_smooth = 0.5 * (bce(out_real_smooth, real_label) + bce(out_fake_smooth, fake_label))

        loss_D = loss_D_sharp + loss_D_smooth
        loss_D.backward()
        opt_Ds.step()

        # ====== --- Train Generator --- ======
        opt_G.zero_grad()
        out_fake_sharp = D_sharp(I_gen_sharp)
        out_fake_smooth = D_smooth(I_gen_smooth)

        gan_loss = 0.5 * (bce(out_fake_sharp, real_label) + bce(out_fake_smooth, real_label))
        recon_loss = l1(I_gen_sharp, I_sharp) + l1(I_gen_smooth, I_smooth)

        loss_G = recon_loss + 0.001 * gan_loss
        loss_G.backward()
        opt_G.step()

        total_G_loss += loss_G.item()
        total_D_loss += loss_D.item()

    print(f"Epoch [{epoch+1}/10]  G_Loss: {total_G_loss/len(loader):.4f}  D_Loss: {total_D_loss/len(loader):.4f}")

torch.save(net, "kernel_estimator_full.pth")
