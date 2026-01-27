import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Switches to a non-interactive backend (no GUI)
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from PSDDataset import PSDDataset
from SplineEstimator import KernelEstimator
from utils import generate_images, robust_normalize, get_scipy_spline, get_torch_spline, radial_mtf_to_2d_otf, compute_gradient_norm, save_checkpoint, load_checkpoint, validate
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import os
import logging
from tqdm import tqdm

# Speed optimization
torch.set_float32_matmul_precision('high')


def train():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_log.txt'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    file_handler = logging.FileHandler('training_log.txt')
    file_handler.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    save_dir = "generated_images_spline_generator_only"
    kernel_dir = "spline_kernels_generator_only"
    checkpoint_dir = "checkpoints_generator_only"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(kernel_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("="*80)
    logger.info(f"Training Configuration - Device: {device}")
    logger.info("Generator-Only Training (No Discriminators)")
    logger.info("="*80)
    
    net = KernelEstimator().to(device)
    l1 = nn.L1Loss().to(device)
    
    # Single optimizer for generator only
    opt_G = torch.optim.Adam(net.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None  # type: ignore
    
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    full_dataset = PSDDataset(
        root_dir=r"D:\Charan work file\KernelEstimator\Data_Root",
        sampling_strategy='all',
        use_ct_windowing=True,
    )
    
    # Split into train and validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Total dataset: {len(full_dataset)} slices")
    logger.info(f"Training set: {len(train_dataset)} slices")
    logger.info(f"Validation set: {len(val_dataset)} slices")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    num_epochs = 100
    latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    best_checkpoint = os.path.join(checkpoint_dir, "best_model.pth")
    
    # Initialize loss history
    start_epoch = 0
    loss_history = {
        'total_loss': [],
        'recon_loss_smooth': [],
        'recon_loss_sharp': [],
        'G_grad_norm': [],
        'val_total_loss': [],
        'val_recon_loss_smooth': [],
        'val_recon_loss_sharp': [],
    }
    best_val_loss = float('inf')
    
    # Uncomment to resume training
    # start_epoch, loss_history = load_checkpoint(
    #     latest_checkpoint, net, opt_G, scaler
    # )
    # if 'val_total_loss' in loss_history and len(loss_history['val_total_loss']) > 0:
    #     best_val_loss = min(loss_history['val_total_loss'])
    
    logger.info(f"Starting from Epoch: {start_epoch + 1}, Total Epochs: {num_epochs}")
    logger.info(f"Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    logger.info(f"Train steps per epoch: {len(train_loader)}, Val steps: {len(val_loader)}")
    logger.info(f"Best validation loss so far: {best_val_loss:.6f}")
    logger.info(f"Loss function: (L1(smooth, gen_smooth) + L1(sharp, gen_sharp)) / 2")
    logger.info("="*80)
    
    for epoch in range(start_epoch, num_epochs):
        # ========== TRAINING ==========
        net.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'recon_loss_smooth': 0.0,
            'recon_loss_sharp': 0.0,
            'G_grad_norm': 0.0,
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")
        
        for batch_idx, (I_smooth, I_sharp, psd_smooth, psd_sharp) in enumerate(pbar):
            I_smooth = I_smooth.to(device, non_blocking=True)
            I_sharp = I_sharp.to(device, non_blocking=True)
            psd_smooth = psd_smooth.to(device, non_blocking=True)
            psd_sharp = psd_sharp.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == "cuda")):  # type: ignore
                knots_smooth, control_smooth = net(psd_smooth)
                knots_sharp, control_sharp = net(psd_sharp)

                mtf_smooth = get_torch_spline(knots_smooth, control_smooth)
                mtf_sharp = get_torch_spline(knots_sharp, control_sharp)
                
                otf_smooth_2d = radial_mtf_to_2d_otf(mtf_smooth, I_smooth.shape[-2:], device)
                otf_sharp_2d = radial_mtf_to_2d_otf(mtf_sharp, I_sharp.shape[-2:], device)

                I_gen_sharp, I_gen_smooth = generate_images(I_smooth, I_sharp, otf_smooth_2d, otf_sharp_2d)
                
                # Calculate loss: (L1(smooth, gen_smooth) + L1(sharp, gen_sharp)) / 2
                loss_smooth = l1(I_gen_smooth, I_smooth)
                loss_sharp = l1(I_gen_sharp, I_sharp)
                total_loss = (loss_smooth + loss_sharp) / 2.0
            
            # Visualization (first batch only)
            if batch_idx == 0:
                with torch.no_grad():
                    n_samples = min(4, I_sharp.size(0))
                    
                    for i in range(n_samples):
                        imgs = torch.stack([
                            robust_normalize(I_smooth[i]),
                            robust_normalize(I_gen_smooth[i]),
                            robust_normalize(I_sharp[i]),
                            robust_normalize(I_gen_sharp[i]),
                        ])
                        save_path = os.path.join(save_dir, f"train_epoch{epoch + 1}_sample{i + 1}.png")
                        save_image(imgs, save_path, nrow=2)
                    
                    # Plot PyTorch MTF curves
                    mtf_smooth_torch = mtf_smooth[0, 0].cpu().numpy()
                    mtf_sharp_torch = mtf_sharp[0, 0].cpu().numpy()
                    
                    num_points = len(mtf_smooth_torch)
                    freq_axis = np.linspace(0, 0.5, num_points)

                    plt.figure(figsize=(10, 6))
                    plt.plot(freq_axis, mtf_smooth_torch, label='Smooth MTF', 
                             color='blue', linewidth=2)
                    plt.plot(freq_axis, mtf_sharp_torch, label='Sharp MTF', 
                             color='red', linewidth=2)
                    
                    plt.title(f'MTF Curves (PyTorch) - Epoch {epoch + 1}')
                    plt.xlabel("Normalized Frequency")
                    plt.ylabel("MTF")
                    plt.xlim(0, 0.5)
                    plt.ylim(0, 1.05)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.savefig(os.path.join(kernel_dir, f'pytorch_mtf_epoch{epoch + 1}.png'), dpi=150)
                    plt.close()

                    # Plot SciPy curves
                    _, curve_smooth_scipy = get_scipy_spline(knots_smooth[0], control_smooth[0])
                    _, curve_sharp_scipy = get_scipy_spline(knots_sharp[0], control_sharp[0])
                    
                    num_points_scipy = len(curve_smooth_scipy)
                    freq_axis_scipy = np.linspace(0, 0.5, num_points_scipy)

                    plt.figure(figsize=(10, 6))
                    plt.plot(freq_axis_scipy, curve_smooth_scipy, label='Smooth MTF (SciPy)', 
                             color='blue', linestyle='--', linewidth=1.5)
                    plt.plot(freq_axis_scipy, curve_sharp_scipy, label='Sharp MTF (SciPy)', 
                             color='red', linestyle='--', linewidth=1.5)
                    
                    plt.title(f'MTF Curves (SciPy Validation) - Epoch {epoch + 1}')
                    plt.xlabel("Normalized Frequency")
                    plt.ylabel("MTF")
                    plt.xlim(0, 0.5)
                    plt.ylim(0, 1.05)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.savefig(os.path.join(kernel_dir, f'scipy_mtf_epoch{epoch + 1}.png'), dpi=150)
                    plt.close()
                    
                    # Calculate difference
                    diff_smooth = mtf_smooth_torch - curve_smooth_scipy
                    diff_sharp = mtf_sharp_torch - curve_sharp_scipy
                    
                    logger.info(f"  MTF smooth range: [{mtf_smooth.min():.6f}, {mtf_smooth.max():.6f}]")
                    logger.info(f"  MTF sharp range: [{mtf_sharp.min():.6f}, {mtf_sharp.max():.6f}]")
                    logger.info(f"  PyTorch vs SciPy max diff (smooth): {np.abs(diff_smooth).max():.6f}")
                    logger.info(f"  PyTorch vs SciPy max diff (sharp): {np.abs(diff_sharp).max():.6f}")
            
            # Train Generator
            opt_G.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(opt_G)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(opt_G)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                opt_G.step()

            G_grad_norm = compute_gradient_norm(net)
            
            epoch_metrics['total_loss'] += total_loss.item()
            epoch_metrics['recon_loss_smooth'] += loss_smooth.item()
            epoch_metrics['recon_loss_sharp'] += loss_sharp.item()
            epoch_metrics['G_grad_norm'] += G_grad_norm

            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Smooth': f"{loss_smooth.item():.4f}",
                'Sharp': f"{loss_sharp.item():.4f}"
            })
        
        # Average training metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)
        
        # ========== VALIDATION ==========
        val_metrics = validate(net, val_loader, l1, device, epoch, save_dir)
        
        # Update loss history
        loss_history['total_loss'].append(epoch_metrics['total_loss'])
        loss_history['recon_loss_smooth'].append(epoch_metrics['recon_loss_smooth'])
        loss_history['recon_loss_sharp'].append(epoch_metrics['recon_loss_sharp'])
        loss_history['G_grad_norm'].append(epoch_metrics['G_grad_norm'])
        
        loss_history['val_total_loss'].append(val_metrics['total_loss'])
        loss_history['val_recon_loss_smooth'].append(val_metrics['recon_loss_smooth'])
        loss_history['val_recon_loss_sharp'].append(val_metrics['recon_loss_sharp'])
        
        # Logging
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - TRAIN")
        logger.info(f"  Total Loss: {epoch_metrics['total_loss']:.4f}")
        logger.info(f"  Smooth Loss: {epoch_metrics['recon_loss_smooth']:.4f} | Sharp Loss: {epoch_metrics['recon_loss_sharp']:.4f}")
        logger.info(f"  Grad Norm: {epoch_metrics['G_grad_norm']:.4f}")
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - VALIDATION")
        logger.info(f"  Total Loss: {val_metrics['total_loss']:.4f}")
        logger.info(f"  Smooth Loss: {val_metrics['recon_loss_smooth']:.4f} | Sharp Loss: {val_metrics['recon_loss_sharp']:.4f}")
        logger.info("-"*80)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_metrics['total_loss']:.4f} | Val Loss: {val_metrics['total_loss']:.4f}")
        
        # Save regular checkpoint (every epoch)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        save_checkpoint(epoch, net, opt_G, scaler, loss_history, checkpoint_path)
        logger.info(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save latest checkpoint (overwrites)
        save_checkpoint(epoch, net, opt_G, scaler, loss_history, latest_checkpoint)
        
        # Save best model based on validation total loss
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            save_checkpoint(epoch, net, opt_G, scaler, loss_history, best_checkpoint)
            logger.info(f"  *** NEW BEST MODEL *** Val Loss: {best_val_loss:.4f}")
            print(f"  *** NEW BEST MODEL *** Val Loss: {best_val_loss:.4f}")

    logger.info("="*80)
    logger.info("Training Completed Successfully!")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info("="*80)

    return net, loss_history


if __name__ == "__main__":
    net, loss_history = train()
    
    # Plot all tracked metrics including validation
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    epochs_x = range(1, len(loss_history['total_loss']) + 1)

    # Plot 1: Total Loss (Train vs Val)
    axes[0, 0].plot(epochs_x, loss_history['total_loss'], label='Train Total Loss', color='blue')
    axes[0, 0].plot(epochs_x, loss_history['val_total_loss'], label='Val Total Loss', color='blue', linestyle='--')
    axes[0, 0].set_title('Total Loss (Train vs Val)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Smooth Loss (Train vs Val)
    axes[0, 1].plot(epochs_x, loss_history['recon_loss_smooth'], label='Train Smooth Loss', color='green')
    axes[0, 1].plot(epochs_x, loss_history['val_recon_loss_smooth'], label='Val Smooth Loss', color='green', linestyle='--')
    axes[0, 1].set_title('Smooth Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Sharp Loss (Train vs Val)
    axes[0, 2].plot(epochs_x, loss_history['recon_loss_sharp'], label='Train Sharp Loss', color='red')
    axes[0, 2].plot(epochs_x, loss_history['val_recon_loss_sharp'], label='Val Sharp Loss', color='red', linestyle='--')
    axes[0, 2].set_title('Sharp Reconstruction Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Gradient Norm
    axes[1, 0].plot(epochs_x, loss_history['G_grad_norm'], label='Generator Grad Norm', color='purple')
    axes[1, 0].set_title('Generator Gradient Norm')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Norm')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Best Model Indicator
    best_epoch = np.argmin(loss_history['val_total_loss']) + 1
    axes[1, 1].axvline(x=best_epoch, color='r', linestyle='--', linewidth=2, label=f'Best Epoch: {best_epoch}')
    axes[1, 1].plot(epochs_x, loss_history['val_total_loss'], label='Val Total Loss', color='green')
    axes[1, 1].set_title('Best Model Indicator')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Val Total Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Summary Statistics
    axes[1, 2].axis('off')
    summary_text = f"""
    Training Summary:
    
    Total Epochs: {len(epochs_x)}
    
    Best Val Epoch: {best_epoch}
    Best Val Loss: {min(loss_history['val_total_loss']):.4f}
    
    Final Train Loss: {loss_history['total_loss'][-1]:.4f}
    Final Val Loss: {loss_history['val_total_loss'][-1]:.4f}
    
    Final Smooth Loss:
      Train: {loss_history['recon_loss_smooth'][-1]:.4f}
      Val: {loss_history['val_recon_loss_smooth'][-1]:.4f}
    
    Final Sharp Loss:
      Train: {loss_history['recon_loss_sharp'][-1]:.4f}
      Val: {loss_history['val_recon_loss_sharp'][-1]:.4f}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    plt.savefig('training_metrics_final_generator_only.png', dpi=150)
    plt.close()
    
    # Save final model (separate from best)
    torch.save(net.state_dict(), "kernel_estimator_final_generator_only.pth")
    print("Training completed!")
    print(f"Best model saved at epoch {best_epoch} with validation loss: {min(loss_history['val_total_loss']):.4f}")
