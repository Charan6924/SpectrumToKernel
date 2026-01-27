import torch
import matplotlib.pyplot as plt
import numpy as np

def robust_normalize(img, lower_percentile=1, upper_percentile=99):
    """
    Percentile-based normalization - more robust to outliers than min-max
    """
    p_low = torch.quantile(img, lower_percentile / 100.0)
    p_high = torch.quantile(img, upper_percentile / 100.0)
    img_norm = (img - p_low) / (p_high - p_low + 1e-8)
    return torch.clamp(img_norm, 0, 1)

def generate_images_percentile(I_smooth, I_sharp, mtf_smooth, mtf_sharp, epsilon=0.01):
    """
    Version with percentile normalization and debugging
    """
    B, _, H, W = I_smooth.shape
    
    print(f"Input I_smooth range: [{I_smooth.min():.4f}, {I_smooth.max():.4f}]")
    print(f"Input I_sharp range: [{I_sharp.min():.4f}, {I_sharp.max():.4f}]")
    print(f"MTF smooth range: [{mtf_smooth.min():.4f}, {mtf_smooth.max():.4f}]")
    print(f"MTF sharp range: [{mtf_sharp.min():.4f}, {mtf_sharp.max():.4f}]")
    
    # FFT with fftshift
    F_smooth = torch.fft.fftshift(torch.fft.fft2(I_smooth))
    F_sharp = torch.fft.fftshift(torch.fft.fft2(I_sharp))

    # Wiener-style regularized deconvolution
    H_s2h = (mtf_sharp * mtf_smooth) / (mtf_smooth**2 + epsilon)
    H_h2s = (mtf_smooth * mtf_sharp) / (mtf_sharp**2 + epsilon)
    
    print(f"Transfer H_s2h range: [{H_s2h.min():.4f}, {H_s2h.max():.4f}]")
    print(f"Transfer H_h2s range: [{H_h2s.min():.4f}, {H_h2s.max():.4f}]")
    
    # Apply transfer functions
    F_gen_sharp = F_smooth * H_s2h
    F_gen_smooth = F_sharp * H_h2s

    # IFFT back to spatial domain
    I_gen_sharp = torch.real(torch.fft.ifft2(torch.fft.ifftshift(F_gen_sharp)))
    I_gen_smooth = torch.real(torch.fft.ifft2(torch.fft.ifftshift(F_gen_smooth)))

    print(f"Before norm - gen_sharp range: [{I_gen_sharp.min():.4f}, {I_gen_sharp.max():.4f}]")
    print(f"Before norm - gen_smooth range: [{I_gen_smooth.min():.4f}, {I_gen_smooth.max():.4f}]")
    
    # Percentile-based normalization (robust to outliers)
    I_gen_sharp = robust_normalize(I_gen_sharp, lower_percentile=1, upper_percentile=99)
    I_gen_smooth = robust_normalize(I_gen_smooth, lower_percentile=1, upper_percentile=99)
    
    print(f"After percentile norm - gen_sharp range: [{I_gen_sharp.min():.4f}, {I_gen_sharp.max():.4f}]")
    print(f"After percentile norm - gen_smooth range: [{I_gen_smooth.min():.4f}, {I_gen_smooth.max():.4f}]")

    # Average over batch dimension
    I_gen_sharp = I_gen_sharp.mean(dim=0, keepdim=True)
    I_gen_smooth = I_gen_smooth.mean(dim=0, keepdim=True)

    return I_gen_sharp, I_gen_smooth


def generate_images_no_norm(I_smooth, I_sharp, mtf_smooth, mtf_sharp, epsilon=0.01):
    """
    Version WITHOUT normalization for comparison
    """
    B, _, H, W = I_smooth.shape
    
    # FFT with fftshift
    F_smooth = torch.fft.fftshift(torch.fft.fft2(I_smooth))
    F_sharp = torch.fft.fftshift(torch.fft.fft2(I_sharp))

    # Wiener-style regularized deconvolution
    H_s2h = (mtf_sharp * mtf_smooth) / (mtf_smooth**2 + epsilon)
    H_h2s = (mtf_smooth * mtf_sharp) / (mtf_sharp**2 + epsilon)
    
    # Apply transfer functions
    F_gen_sharp = F_smooth * H_s2h
    F_gen_smooth = F_sharp * H_h2s

    # IFFT back to spatial domain
    I_gen_sharp = torch.real(torch.fft.ifft2(torch.fft.ifftshift(F_gen_sharp)))
    I_gen_smooth = torch.real(torch.fft.ifft2(torch.fft.ifftshift(F_gen_smooth)))

    print(f"No norm - gen_sharp range: [{I_gen_sharp.min():.4f}, {I_gen_sharp.max():.4f}]")
    print(f"No norm - gen_smooth range: [{I_gen_smooth.min():.4f}, {I_gen_smooth.max():.4f}]")
    
    # Just clip to [0, 1]
    I_gen_sharp = torch.clamp(I_gen_sharp, 0, 1)
    I_gen_smooth = torch.clamp(I_gen_smooth, 0, 1)

    # Average over batch dimension
    I_gen_sharp = I_gen_sharp.mean(dim=0, keepdim=True)
    I_gen_smooth = I_gen_smooth.mean(dim=0, keepdim=True)

    return I_gen_sharp, I_gen_smooth


def create_synthetic_data(batch_size=4, size=256):
    """
    Create synthetic test data
    """
    # Create simple smooth and sharp test images
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Create a simple pattern
    pattern = torch.sin(xx * 5) * torch.cos(yy * 5) * 0.5 + 0.5
    
    I_smooth = pattern.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    I_sharp = pattern.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # Add some noise
    I_smooth = I_smooth + torch.randn_like(I_smooth) * 0.05
    I_sharp = I_sharp + torch.randn_like(I_sharp) * 0.02
    
    # Clip to [0, 1]
    I_smooth = torch.clamp(I_smooth, 0, 1)
    I_sharp = torch.clamp(I_sharp, 0, 1)
    
    # Create synthetic MTFs (gaussian-like)
    freq_x = torch.fft.fftfreq(size)
    freq_y = torch.fft.fftfreq(size)
    freq_y, freq_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
    freq_radius = torch.sqrt(freq_x**2 + freq_y**2)
    
    # Smooth MTF - drops off faster
    mtf_smooth = torch.exp(-freq_radius**2 / (2 * 0.1**2))
    mtf_smooth = mtf_smooth.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # Sharp MTF - drops off slower
    mtf_sharp = torch.exp(-freq_radius**2 / (2 * 0.3**2))
    mtf_sharp = mtf_sharp.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    return I_smooth, I_sharp, mtf_smooth, mtf_sharp


def visualize_results(I_smooth, I_sharp, I_gen_sharp_percentile, I_gen_smooth_percentile,
                      I_gen_sharp_no_norm, I_gen_smooth_no_norm):
    """
    Visualize all results side by side
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Original and Percentile Normalized
    axes[0, 0].imshow(I_smooth[0, 0].cpu(), cmap='gray')
    axes[0, 0].set_title('Input: Smooth')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(I_sharp[0, 0].cpu(), cmap='gray')
    axes[0, 1].set_title('Input: Sharp')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(I_gen_sharp_percentile[0, 0].cpu(), cmap='gray')
    axes[0, 2].set_title('Generated Sharp (Percentile Norm)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(I_gen_smooth_percentile[0, 0].cpu(), cmap='gray')
    axes[0, 3].set_title('Generated Smooth (Percentile Norm)')
    axes[0, 3].axis('off')
    
    # Row 2: No normalization results
    axes[1, 0].imshow(I_smooth[0, 0].cpu(), cmap='gray')
    axes[1, 0].set_title('Input: Smooth')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(I_sharp[0, 0].cpu(), cmap='gray')
    axes[1, 1].set_title('Input: Sharp')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(I_gen_sharp_no_norm[0, 0].cpu(), cmap='gray')
    axes[1, 2].set_title('Generated Sharp (No Norm)')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(I_gen_smooth_no_norm[0, 0].cpu(), cmap='gray')
    axes[1, 3].set_title('Generated Smooth (No Norm)')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('percentile_norm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved comparison to 'percentile_norm_comparison.png'")


def main():
    """
    Main test script
    """
    print("="*80)
    print("Testing Percentile Normalization vs No Normalization")
    print("="*80)
    
    # Create synthetic test data
    print("\nCreating synthetic test data...")
    I_smooth, I_sharp, mtf_smooth, mtf_sharp = create_synthetic_data(batch_size=4, size=256)
    
    # Test with different epsilon values
    epsilon_values = [0.001, 0.01, 0.1]
    
    for eps in epsilon_values:
        print(f"\n{'='*80}")
        print(f"Testing with epsilon = {eps}")
        print(f"{'='*80}")
        
        print("\n--- WITH PERCENTILE NORMALIZATION ---")
        I_gen_sharp_perc, I_gen_smooth_perc = generate_images_percentile(
            I_smooth, I_sharp, mtf_smooth, mtf_sharp, epsilon=eps
        )
        
        print("\n--- WITHOUT NORMALIZATION (just clipping) ---")
        I_gen_sharp_no, I_gen_smooth_no = generate_images_no_norm(
            I_smooth, I_sharp, mtf_smooth, mtf_sharp, epsilon=eps
        )
        
        # Visualize results
        visualize_results(I_smooth, I_sharp, I_gen_sharp_perc, I_gen_smooth_perc,
                         I_gen_sharp_no, I_gen_smooth_no)


if __name__ == "__main__":
    main()