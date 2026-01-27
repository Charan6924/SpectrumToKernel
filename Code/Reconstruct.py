import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import matplotlib.pyplot as plt  # Added for visualization
import os  # Added for path manipulation
from utils import cox_de_boor,get_torch_spline,get_scipy_spline,profile_to_kernel, apply_window
from SplineEstimator import KernelEstimator
from utils import Preprocessor
from utils import profile_to_kernel


test_scan = r"D:\Charan work file\KernelEstimator\Data_Root\trainB\0B14X41758_filter_E.nii"

def get_dummy_gaussian_kernel(kernel_size=31, sigma=1.5, device='cuda'):
    """
    Creates a standard 2D Gaussian kernel.
    sigma: Controls the width. 
           1.0 = sharp/small blur
           3.0 = wide/strong blur
    """
    range_vec = torch.arange(-(kernel_size//2), (kernel_size//2) + 1, device=device)
    y, x = torch.meshgrid(range_vec, range_vec, indexing='ij')
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    return kernel
def reconstruct_nifti(nii_path, model, output_path, device='cuda', save_debug_images=True): 
    print(f"Loading {nii_path}...")
    img_obj = nib.load(nii_path)
    vol_data = img_obj.get_fdata()
    affine = img_obj.affine
    h, w, d = vol_data.shape
    debug_slice_idx = d // 2 # Calculate middle slice for debugging
    preproc = Preprocessor(shape=(h, w))
    reconstructed_vol = np.zeros_like(vol_data)
    model.eval()
    model.to(device)
    print(f"Starting reconstruction on {d} slices...")
    K_wiener = 0.01
    
    with torch.no_grad():
        for i in range(d):
            slice_img = vol_data[:, :, i]
            psd_profile = preproc.process_slice(slice_img)
            psd_tensor = torch.from_numpy(psd_profile).float().to(device)
            psd_tensor = psd_tensor.view(1, 1, 363)
            knots, controls = model(psd_tensor)
            profile_curve = get_torch_spline(knots, controls, num_points=363)
            kernel_2d = kernel_2d = profile_to_kernel(profile_curve, kernel_size=31)
            # kernel_2d = get_dummy_gaussian_kernel(kernel_size=31, sigma=1.5, device=device)
            kernel_2d = apply_window(kernel_2d, device)
            kernel_2d = kernel_2d / (kernel_2d.sum() + 1e-8)

            if save_debug_images and i == debug_slice_idx:
                print(f"Saving debug visualization for Slice {i}...")
                k_cpu = kernel_2d.squeeze().cpu().numpy()
                p_cpu = profile_curve.squeeze().cpu().numpy()
                
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(k_cpu, cmap='inferno')
                plt.colorbar()
                plt.title(f"Inference Kernel (Slice {i})")
                
                plt.subplot(1, 2, 2)
                plt.plot(p_cpu, color='blue', linewidth=2)
                plt.grid(True, alpha=0.3)
                plt.title("Radial Profile")
                
                debug_path = output_path.replace(".nii", "_2d_kernel.png")
                plt.savefig(debug_path)
                plt.close()
                print(f"-> Saved kernel viz to {debug_path}")

            img_t = torch.tensor(slice_img, dtype=torch.float32, device=device)
            pad_h = h - 31
            pad_w = w - 31
            kernel_padded = F.pad(kernel_2d, (0, pad_w, 0, pad_h))
            shift = -(31 // 2)
            kernel_padded = torch.roll(kernel_padded, shifts=(shift, shift), dims=(0, 1))
            Y = torch.fft.rfft2(img_t)
            H = torch.fft.rfft2(kernel_padded)
            X_fft = (Y * torch.conj(H)) / (torch.abs(H)**2 + K_wiener)
            restored = torch.fft.irfft2(X_fft, s=(h, w))
            reconstructed_vol[:, :, i] = restored.cpu().numpy()
            
            if i % 50 == 0:
                print(f"Processed slice {i}/{d}")

    new_img = nib.Nifti1Image(reconstructed_vol, affine)
    nib.save(new_img, output_path)
    print(f"Saved to {output_path}")

device = 'cuda'
loaded_model = torch.load(r'D:\Charan work file\KernelEstimator\kernel_estimator_full.pth', weights_only=False)
loaded_model.eval()
reconstruct_nifti(test_scan, loaded_model, 'reconstructed_scan.nii', device=device)