import torch
import numpy as np
import matplotlib.pyplot as plt
from PSDDataset import PSDDataset
from torch.utils.data import DataLoader

def diagnose_data(dataset_path, num_samples=10):
    """
    Diagnostic tool to check if smooth and sharp PSDs are actually different
    """
    print("="*80)
    print("DIAGNOSTIC: Checking PSD Data Quality")
    print("="*80)
    
    dataset = PSDDataset(
        root_dir=dataset_path,
        sampling_strategy='uniform',
        slices_per_volume=5,
        use_ct_windowing=True,
    )
    
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    
    # Get one batch
    I_smooth, I_sharp, psd_smooth, psd_sharp = next(iter(loader))
    
    # Statistics
    print(f"\nBatch size: {I_smooth.shape[0]}")
    print(f"\nImage statistics:")
    print(f"  I_smooth: min={I_smooth.min():.6f}, max={I_smooth.max():.6f}, mean={I_smooth.mean():.6f}")
    print(f"  I_sharp:  min={I_sharp.min():.6f}, max={I_sharp.max():.6f}, mean={I_sharp.mean():.6f}")
    
    print(f"\nPSD statistics:")
    print(f"  psd_smooth: min={psd_smooth.min():.6f}, max={psd_smooth.max():.6f}, mean={psd_smooth.mean():.6f}")
    print(f"  psd_sharp:  min={psd_sharp.min():.6f}, max={psd_sharp.max():.6f}, mean={psd_sharp.mean():.6f}")
    
    # Calculate differences
    psd_diff = (psd_smooth - psd_sharp).abs()
    print(f"\nPSD differences:")
    print(f"  Mean abs diff: {psd_diff.mean():.6f}")
    print(f"  Max abs diff:  {psd_diff.max():.6f}")
    print(f"  Min abs diff:  {psd_diff.min():.6f}")
    
    # Check if PSDs are too similar
    similarity = 1.0 - (psd_diff.mean() / (psd_smooth.abs().mean() + 1e-8))
    print(f"  Similarity score: {similarity:.4f} (1.0 = identical, 0.0 = completely different)")
    
    if similarity > 0.95:
        print("\n⚠️  WARNING: PSDs are very similar! Network may struggle to learn differences.")
    elif similarity > 0.90:
        print("\n⚠️  CAUTION: PSDs are quite similar. Consider increasing contrast between smooth/sharp.")
    else:
        print("\n✓ PSDs show reasonable differences.")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot example images
    axes[0, 0].imshow(I_smooth[0, 0].numpy(), cmap='gray')
    axes[0, 0].set_title('Smooth Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(I_sharp[0, 0].numpy(), cmap='gray')
    axes[0, 1].set_title('Sharp Image')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow((I_sharp[0, 0] - I_smooth[0, 0]).numpy(), cmap='seismic')
    axes[0, 2].set_title('Difference (Sharp - Smooth)')
    axes[0, 2].axis('off')
    
    # Plot PSDs
    freq_axis = np.arange(psd_smooth.shape[-1])
    for i in range(min(3, num_samples)):
        axes[1, 0].plot(freq_axis, psd_smooth[i, 0].numpy(), alpha=0.6, label=f'Sample {i+1}')
    axes[1, 0].set_title('Smooth PSDs')
    axes[1, 0].set_xlabel('Frequency Index')
    axes[1, 0].set_ylabel('Log PSD')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    for i in range(min(3, num_samples)):
        axes[1, 1].plot(freq_axis, psd_sharp[i, 0].numpy(), alpha=0.6, label=f'Sample {i+1}')
    axes[1, 1].set_title('Sharp PSDs')
    axes[1, 1].set_xlabel('Frequency Index')
    axes[1, 1].set_ylabel('Log PSD')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot average PSDs with difference
    avg_smooth = psd_smooth.mean(dim=0)[0].numpy()
    avg_sharp = psd_sharp.mean(dim=0)[0].numpy()
    axes[1, 2].plot(freq_axis, avg_smooth, label='Smooth (avg)', linewidth=2)
    axes[1, 2].plot(freq_axis, avg_sharp, label='Sharp (avg)', linewidth=2)
    axes[1, 2].fill_between(freq_axis, avg_smooth, avg_sharp, alpha=0.3, label='Difference')
    axes[1, 2].set_title('Average PSDs Comparison')
    axes[1, 2].set_xlabel('Frequency Index')
    axes[1, 2].set_ylabel('Log PSD')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('psd_diagnostic.png', dpi=150)
    print(f"\n✓ Diagnostic plot saved to: psd_diagnostic.png")
    plt.close()
    
    return dataset, loader


def diagnose_model_outputs(net, loader, device='cuda'):
    """
    Check if model produces different outputs for smooth vs sharp PSDs
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC: Checking Model Output Diversity")
    print("="*80)
    
    net.eval()
    with torch.no_grad():
        I_smooth, I_sharp, psd_smooth, psd_sharp = next(iter(loader))
        I_smooth = I_smooth.to(device)
        I_sharp = I_sharp.to(device)
        psd_smooth = psd_smooth.to(device)
        psd_sharp = psd_sharp.to(device)
        
        # Forward pass
        knots_smooth, control_smooth = net(psd_smooth)
        knots_sharp, control_sharp = net(psd_sharp)
        
        print(f"\nModel outputs:")
        print(f"  Knots smooth:  shape={knots_smooth.shape}, range=[{knots_smooth.min():.6f}, {knots_smooth.max():.6f}]")
        print(f"  Knots sharp:   shape={knots_sharp.shape}, range=[{knots_sharp.min():.6f}, {knots_sharp.max():.6f}]")
        print(f"  Control smooth: shape={control_smooth.shape}, range=[{control_smooth.min():.6f}, {control_smooth.max():.6f}]")
        print(f"  Control sharp:  shape={control_sharp.shape}, range=[{control_sharp.min():.6f}, {control_sharp.max():.6f}]")
        
        # Check differences
        knot_diff = (knots_smooth - knots_sharp).abs().mean()
        control_diff = (control_smooth - control_sharp).abs().mean()
        
        print(f"\nDifferences:")
        print(f"  Mean abs knot difference:    {knot_diff:.6f}")
        print(f"  Mean abs control difference: {control_diff:.6f}")
        
        if control_diff < 0.01:
            print("\n⚠️  CRITICAL: Control points are nearly identical!")
            print("    The model is not learning to distinguish smooth from sharp.")
            print("    Possible causes:")
            print("    1. PSDs are too similar (check data diagnostic above)")
            print("    2. Model architecture is too constrained")
            print("    3. Learning rate too low or training not converged")
            print("    4. Loss function doesn't encourage differentiation")
        elif control_diff < 0.05:
            print("\n⚠️  WARNING: Control points are very similar.")
            print("    Consider adding a diversity loss term.")
        else:
            print("\n✓ Control points show reasonable differences.")
        
        # Visualize control points
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot first 5 examples
        for i in range(min(5, control_smooth.shape[0])):
            axes[0].plot(control_smooth[i].cpu().numpy(), alpha=0.6, label=f'Sample {i+1}')
        axes[0].set_title('Smooth Control Points')
        axes[0].set_xlabel('Control Point Index')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        for i in range(min(5, control_sharp.shape[0])):
            axes[1].plot(control_sharp[i].cpu().numpy(), alpha=0.6, label=f'Sample {i+1}')
        axes[1].set_title('Sharp Control Points')
        axes[1].set_xlabel('Control Point Index')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('control_points_diagnostic.png', dpi=150)
        print(f"\n✓ Control points plot saved to: control_points_diagnostic.png")
        plt.close()


if __name__ == "__main__":
    # Run diagnostics
    dataset_path = r"D:\Charan work file\KernelEstimator\Data_Root"
    
    print("Running data quality diagnostic...")
    dataset, loader = diagnose_data(dataset_path, num_samples=10)
    
    # Uncomment to check model outputs (after training has started)
    print("\nRunning model output diagnostic...")
    from SplineEstimator import KernelEstimator
    net = KernelEstimator().to('cuda')
    checkpoint = torch.load(r'D:\Charan work file\KernelEstimator\checkpoints_generator_only\checkpoint_epoch_98.pth', map_location='cuda')
    net.load_state_dict(checkpoint['net_state_dict'])
    diagnose_model_outputs(net, loader, device='cuda')
    
    print("\n" + "="*80)
    print("Diagnostic complete!")
    print("="*80)