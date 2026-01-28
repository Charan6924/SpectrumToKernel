import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from scipy.ndimage import sobel
from utils import cox_de_boor, get_torch_spline, get_scipy_spline, profile_to_kernel, apply_window, radial_mtf_to_2d_otf_fixed
from SplineEstimator import KernelEstimator
from utils import Preprocessor
from utils import profile_to_kernel, load_checkpoint_for_inference


def reconstruct_nifti(nii_path, model, output_path, device='cuda', save_debug_images=True,
                               max_amplification=2.0,
                               preserve_range=True,
                               noise_suppression=True,
                               wiener_k=0.008,
                               frequency_rolloff=0.8):
    print(f"Loading {nii_path}...")
    img_obj = nib.load(nii_path) #type: ignore
    vol_data = img_obj.get_fdata() #type: ignore
    affine = img_obj.affine #type: ignore
    h, w, d = vol_data.shape
    debug_slice_idx = d // 2
    
    preproc = Preprocessor(shape=(h, w))
    reconstructed_vol = np.zeros_like(vol_data)
    
    model.eval()
    model.to(device)
    
    print(f"Starting reconstruction on {d} slices...")
    
    epsilon = 0.001 
    
    with torch.no_grad():
        for i in range(d):
            slice_img = vol_data[:, :, i]
            
            if slice_img.max() == 0:
                reconstructed_vol[:, :, i] = slice_img
                continue
            
            # Global stats for restoration
            orig_mean = slice_img.mean()
            orig_std = slice_img.std()
            orig_min = slice_img.min()
            orig_max = slice_img.max()
            
            # Local noise estimation
            air_mask = slice_img < -900
            noise_std = slice_img[air_mask].std() if air_mask.sum() > 100 else orig_std * 0.1
                
            # MTF Estimation via Spline Model
            psd_blurred = preproc.process_slice(slice_img)
            psd_tensor = torch.from_numpy(psd_blurred).float().to(device).view(1, 1, 363)
            
            knots_blurred, controls_blurred = model(psd_tensor)
            mtf_blurred = get_torch_spline(knots_blurred, controls_blurred, num_points=363)
            otf_blurred_2d = radial_mtf_to_2d_otf_fixed(mtf_blurred, (h, w), device).squeeze()
            
            # Target Sharp MTF (Gaussian)
            radius = torch.linspace(0, 1, 363, device=device)
            sigma = 0.38
            mtf_sharp_profile = torch.exp(-(radius ** 2) / (2 * sigma ** 2)).view(1, 1, -1)
            otf_sharp_2d = radial_mtf_to_2d_otf_fixed(mtf_sharp_profile, (h, w), device).squeeze()
            
            # Transfer Function with Frequency Rolloff Mask
            OTF_transfer = otf_sharp_2d / (otf_blurred_2d + epsilon)
            freq_y = torch.fft.fftfreq(h, device=device).view(-1, 1)
            freq_x = torch.fft.fftfreq(w, device=device).view(1, -1)
            freq_radius_norm = torch.sqrt(freq_y**2 + freq_x**2)
            freq_radius_norm /= freq_radius_norm.max()
            
            rolloff_mask = torch.where(
                freq_radius_norm < frequency_rolloff,
                torch.ones_like(freq_radius_norm),
                torch.exp(-((freq_radius_norm - frequency_rolloff) / (1 - frequency_rolloff))**2 * 5)
            )
            
            OTF_transfer = torch.clamp(OTF_transfer * rolloff_mask, min=0.5, max=max_amplification)
            
            # Wiener Noise Suppression
            if noise_suppression:
                img_t = torch.tensor(slice_img, dtype=torch.float32, device=device)
                F_blurred = torch.fft.fft2(img_t)
                wiener_filter = (torch.abs(F_blurred)**2) / (torch.abs(F_blurred)**2 + wiener_k * (noise_std**2))
                OTF_transfer *= wiener_filter
            
            # Frequency Domain Restoration
            F_blurred = torch.fft.fft2(torch.tensor(slice_img, dtype=torch.float32, device=device))
            F_sharp = F_blurred * OTF_transfer.to(torch.complex64)
            restored_np = torch.real(torch.fft.ifft2(F_sharp)).cpu().numpy()
            
            # Intensity and Mean Preservation
            if preserve_range:
                if noise_std > orig_std * 0.25:
                    restored_np = np.clip(restored_np, *np.percentile(restored_np, [0.1, 99.9]))
                # Match mean exactly while allowing sharpened variance
                restored_np = restored_np + (orig_mean - restored_np.mean())
                restored_np = np.clip(restored_np, orig_min, orig_max)
            
            reconstructed_vol[:, :, i] = restored_np

            if i == debug_slice_idx:
                edges_orig = np.sqrt(sobel(slice_img, 0)**2 + sobel(slice_img, 1)**2)
                edges_rest = np.sqrt(sobel(restored_np, 0)**2 + sobel(restored_np, 1)**2)
                edge_gain = edges_rest.mean() / edges_orig.mean()
                std_ratio = restored_np.std() / orig_std
                mean_err = abs(restored_np.mean() - orig_mean)

                if save_debug_images:
                    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
                    
                    im0 = axes[0].imshow(slice_img, cmap='gray', vmin=orig_min, vmax=orig_max)
                    axes[0].set_title(f"Original: {orig_std:.1f}")
                    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                    
                    im1 = axes[1].imshow(restored_np, cmap='gray', vmin=orig_min, vmax=orig_max)
                    axes[1].set_title(f"Genearated Image")
                    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                    
                    for ax in axes: ax.axis('off')
                    
                    debug_path = output_path.replace(".nii", "_debug.png")
                    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"Saved debug comparison to {debug_path}")

    new_img = nib.Nifti1Image(reconstructed_vol, affine) #type: ignore
    nib.save(new_img, output_path)#type: ignore
    print(f"Full volume saved to {output_path}")


if __name__ == "__main__":
    # Path configuration
    test_scan = r"D:\Charan work file\KernelEstimator\Data_Root\trainB\0B14X41758_filter_E.nii"
    checkpoint = r"D:\Charan work file\KernelEstimator\best_model.pth"
    output_name = 'reconstructed.nii'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize Model
    net = KernelEstimator()
    net = load_checkpoint_for_inference(checkpoint, net)
    reconstruct_nifti(
        test_scan, 
        net, 
        output_name, 
        device=device,
        max_amplification=2.0,
        wiener_k=0.008,
        frequency_rolloff=0.8,
        save_debug_images=True
    )