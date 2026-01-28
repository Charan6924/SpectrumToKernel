import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import BSpline
import os

def radial_average(psd):
    y, x = np.indices(psd.shape)
    center = np.array(psd.shape) / 2.0
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)
    return radial_profile

def generate_images(I_smooth, I_sharp, otf_smooth_2d, otf_sharp_2d, epsilon=0.01):
    """
    Implements the exact formulation from your diagram:
    OTF_smooth2sharp = k_sharp / (k_smooth + epsilon)
    OTF_sharp2smooth = k_smooth / (k_sharp + epsilon)
    I_generated_smooth = F^-1[F[I_sharp] * OTF_sharp2smooth]
    I_generated_sharp = F^-1[F[I_smooth] * OTF_smooth2sharp]
    
    Args:
        I_smooth, I_sharp: [B, 1, H, W] - input images
        otf_smooth_2d, otf_sharp_2d: [B, 1, H, W] - 2D OTF maps (k_smooth, k_sharp)
        epsilon: regularization parameter
    
    Returns:
        I_gen_sharp, I_gen_smooth: [B, 1, H, W] - generated images
    """
    # Take FFT of input images
    F_smooth = torch.fft.fft2(I_smooth)  # F[I_smooth]
    F_sharp = torch.fft.fft2(I_sharp)    # F[I_sharp]
    
    # Compute transfer functions exactly as in diagram
    OTF_smooth2sharp = otf_sharp_2d / (otf_smooth_2d + epsilon)
    OTF_sharp2smooth = otf_smooth_2d / (otf_sharp_2d + epsilon)
    
    # Apply transfer functions in frequency domain
    F_gen_sharp = F_smooth * OTF_smooth2sharp   # F[I_smooth] * OTF_smooth2sharp
    F_gen_smooth = F_sharp * OTF_sharp2smooth   # F[I_sharp] * OTF_sharp2smooth
    
    # Take inverse FFT to get spatial domain images
    I_gen_sharp = torch.real(torch.fft.ifft2(F_gen_sharp))   # F^-1[...]
    I_gen_smooth = torch.real(torch.fft.ifft2(F_gen_smooth)) # F^-1[...]
    
    # Normalize to [0, 1] range
    I_gen_sharp = robust_normalize(I_gen_sharp, lower_percentile=2, upper_percentile=98)
    I_gen_smooth = robust_normalize(I_gen_smooth, lower_percentile=2, upper_percentile=98)
    
    return I_gen_sharp, I_gen_smooth


def robust_normalize(img, lower_percentile=2, upper_percentile=98):
    """
    Percentile-based normalization
    Handles both single images [C, H, W] and batches [B, C, H, W]
    """
    # Handle single image case
    if img.ndim == 3:
        img = img.unsqueeze(0)  # Add batch dimension [1, C, H, W]
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, C, H, W = img.shape
    img_normalized = torch.zeros_like(img)
    
    for b in range(B):
        img_b = img[b]
        p_low = torch.quantile(img_b, lower_percentile / 100.0)
        p_high = torch.quantile(img_b, upper_percentile / 100.0)
        
        if p_high > p_low:
            img_normalized[b] = (img_b - p_low) / (p_high - p_low)
        else:
            img_normalized[b] = img_b * 0.5  # fallback to middle gray
    
    img_normalized = torch.clamp(img_normalized, 0, 1)
    
    # Remove batch dimension if input was single image
    if squeeze_output:
        img_normalized = img_normalized.squeeze(0)
    
    return img_normalized

def get_scipy_spline(knots_tensor, control_tensor, degree=3, num_points=363):

    t = knots_tensor.detach().cpu().numpy().flatten()
    c = control_tensor.detach().cpu().numpy().flatten()

    spl = BSpline(t, c, k=degree)

    x_axis = np.linspace(0, 1, num_points)
    y_axis = spl(x_axis)
    
    return x_axis, y_axis

def cox_de_boor(t, k, knots, degree):
    if degree == 0:
        k_start = knots[:, k].unsqueeze(1)
        k_end = knots[:, k+1].unsqueeze(1)
        mask = (t >= k_start) & (t < k_end)
        return mask.float()

    epsilon = 1e-6
    
    term1_num = (t - knots[:, k].unsqueeze(1))
    term1_den = (knots[:, k+degree].unsqueeze(1) - knots[:, k].unsqueeze(1))
    term1 = (term1_num / (term1_den + epsilon)) * cox_de_boor(t, k, knots, degree-1)
    
    term2_num = (knots[:, k+degree+1].unsqueeze(1) - t)
    term2_den = (knots[:, k+degree+1].unsqueeze(1) - knots[:, k+1].unsqueeze(1))
    term2 = (term2_num / (term2_den + epsilon)) * cox_de_boor(t, k+1, knots, degree-1)
    
    return term1 + term2

def get_torch_spline(knots, control, num_points=363, degree=3):
    batch_size = knots.shape[0]
    device = knots.device
    num_control = control.shape[1]
    
    t_eval = torch.linspace(0, 1, num_points, device=device).unsqueeze(0).repeat(batch_size, 1)
    
    final_curve = torch.zeros(batch_size, num_points, device=device)
    
    for i in range(num_control):
        basis_i = cox_de_boor(t_eval, i, knots, degree)
        final_curve += control[:, i].unsqueeze(1) * basis_i
        
    return final_curve.unsqueeze(1) # Shape [Batch, 1, 363]

class Preprocessor:
    def __init__(self, shape=(512, 512)):
        y, x = np.indices(shape)
        center = np.array(shape) / 2.0
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        self.r = r.astype(np.int32)
        self.nr = np.bincount(self.r.ravel())
        target_len = 363
        if len(self.nr) > target_len:
            self.nr = self.nr[:target_len]
        elif len(self.nr) < target_len:
            pad_size = target_len - len(self.nr)
            self.nr = np.pad(self.nr, (0, pad_size), 'constant', constant_values=1)

    def process_slice(self, slice_img):
        f = np.fft.fft2(slice_img)
        fshift = np.fft.fftshift(f)
        psd = np.abs(fshift) ** 2
        tbin = np.bincount(self.r.ravel(), psd.ravel())
        target_len = 363
        if len(tbin) > target_len:
            tbin = tbin[:target_len]
        elif len(tbin) < target_len:
            tbin = np.pad(tbin, (0, target_len - len(tbin)), 'constant')     
        radial_profile = tbin / (self.nr + 1e-8)
        radial_profile = np.log(radial_profile + 1e-8)
        
        return radial_profile

def profile_to_kernel(profile_1d, kernel_size=31):
    device = profile_1d.device
    range_vec = torch.linspace(-1, 1, kernel_size, device=device)
    y, x = torch.meshgrid(range_vec, range_vec, indexing='ij')
    r = torch.sqrt(x**2 + y**2)
    grid_coords = 2.0 * r - 1.0 
    zeros = torch.zeros_like(grid_coords)
    grid = torch.stack([grid_coords, zeros], dim=-1).unsqueeze(0) # [1, K, K, 2]

    inp = profile_1d.view(1, 1, 1, -1)
    
    kernel = F.grid_sample(inp, grid, align_corners=True, padding_mode='border')
    kernel = kernel.view(kernel_size, kernel_size)
    
    mask = (r <= 1.0).float()
    kernel = kernel * mask
    kernel = kernel / (kernel.sum() + 1e-8)
    
    return kernel

def radial_mtf_to_2d_otf(mtf_1d, image_shape, device):
    """
    Convert 1D radial MTF profile to 2D OTF in frequency space
    
    mtf_1d: [B, 1, L] - radial MTF profile (e.g., 363 points)
    image_shape: (H, W) - target image dimensions
    
    Returns: [B, 1, H, W] - 2D OTF ready for frequency domain operations
    """
    B, C, L = mtf_1d.shape
    H, W = image_shape
    
    # Create frequency space radial distance map (centered)
    y_freq = torch.fft.fftfreq(H, device=device).view(-1, 1)
    x_freq = torch.fft.fftfreq(W, device=device).view(1, -1)
    r_freq = torch.sqrt(y_freq**2 + x_freq**2)  # [H, W]
    
    # Normalize radius to [0, 1] range
    nyquist = 0.5  # Nyquist frequency
    r_norm = (r_freq / nyquist).clamp(0, 1)  # [H, W]
    
    # Map normalized radius to MTF profile indices
    indices = (r_norm * (L - 1)).long().clamp(0, L - 1)  # [H, W]
    
    # Build 2D OTF for each batch
    otf_2d = torch.zeros(B, 1, H, W, device=device)
    for b in range(B):
        otf_2d[b, 0] = mtf_1d[b, 0, indices]
    
    return otf_2d

def apply_window(kernel, device):
    """
    Multiplies the kernel by a 2D Hanning window to force edges to zero.
    This eliminates the "cliff" that causes ringing artifacts.
    """
    k_size = kernel.shape[0]
    # Create 1D window (bell curve shape)
    win1d = torch.hann_window(k_size, periodic=False, device=device)
    # Create 2D window by outer product
    win2d = win1d.unsqueeze(1) * win1d.unsqueeze(0)
    return kernel * win2d

def compute_gradient_norm(model):
    """Compute the L2 norm of gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def save_checkpoint(epoch, net, opt_G, scaler, loss_history, filepath):
    """Save training checkpoint - Generator only version."""
    checkpoint = {
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss_history': loss_history,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, net, opt_G, scaler):
    """Load training checkpoint - Generator only version."""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        net.load_state_dict(checkpoint['net_state_dict'])
        opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        if scaler and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint['epoch'] + 1, checkpoint['loss_history']
    else:
        return 0, {
            'total_loss': [],
            'recon_loss_smooth': [],
            'recon_loss_sharp': [],
            'G_grad_norm': [],
        }
def load_checkpoint_for_inference(filepath, net):
    """Load model weights for inference only."""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location='cpu')  # or 'cuda'
        net.load_state_dict(checkpoint['net_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    net.eval()  # Set to evaluation mode
    return net

def validate(net, val_loader, l1, device, epoch, save_dir):
    """
    Validation function - Generator only version
    """
    net.eval()
    
    val_metrics = {
        'total_loss': 0.0,
        'recon_loss_smooth': 0.0,
        'recon_loss_sharp': 0.0,
    }
    
    with torch.no_grad():
        for batch_idx, (I_smooth, I_sharp, psd_smooth, psd_sharp) in enumerate(val_loader):
            I_smooth = I_smooth.to(device, non_blocking=True)
            I_sharp = I_sharp.to(device, non_blocking=True)
            psd_smooth = psd_smooth.to(device, non_blocking=True)
            psd_sharp = psd_sharp.to(device, non_blocking=True)
            
            # Forward pass
            knots_smooth, control_smooth = net(psd_smooth)
            knots_sharp, control_sharp = net(psd_sharp)

            mtf_smooth = get_torch_spline(knots_smooth, control_smooth)
            mtf_sharp = get_torch_spline(knots_sharp, control_sharp)
            
            otf_smooth_2d = radial_mtf_to_2d_otf(mtf_smooth, I_smooth.shape[-2:], device)
            otf_sharp_2d = radial_mtf_to_2d_otf(mtf_sharp, I_sharp.shape[-2:], device)

            I_gen_sharp, I_gen_smooth = generate_images(I_smooth, I_sharp, otf_smooth_2d, otf_sharp_2d)
            
            # Calculate losses
            loss_smooth = l1(I_gen_smooth, I_smooth)
            loss_sharp = l1(I_gen_sharp, I_sharp)
            
            # Total loss: (loss_smooth + loss_sharp) / 2
            total_loss = (loss_smooth + loss_sharp) / 2.0
            
            # Accumulate metrics
            val_metrics['total_loss'] += total_loss.item()
            val_metrics['recon_loss_smooth'] += loss_smooth.item()
            val_metrics['recon_loss_sharp'] += loss_sharp.item()
    
    # Average over all batches
    for key in val_metrics:
        val_metrics[key] /= len(val_loader)
    
    return val_metrics

def radial_mtf_to_2d_otf_fixed(mtf_1d, image_shape, device):
    """
    Convert 1D radial MTF profile to 2D OTF in frequency space
    
    mtf_1d: [B, 1, L] - radial MTF profile (e.g., 363 points)
    image_shape: (H, W) - target image dimensions
    
    Returns: [B, 1, H, W] - 2D OTF ready for FFT operations
    """
    B, C, L = mtf_1d.shape
    H, W = image_shape
    
    # Create frequency space coordinate grid (non-shifted, as FFT expects)
    y_freq = torch.fft.fftfreq(H, device=device).view(-1, 1)  # [-0.5, 0.5]
    x_freq = torch.fft.fftfreq(W, device=device).view(1, -1)
    
    # Compute radial frequency
    r_freq = torch.sqrt(y_freq**2 + x_freq**2)  # [H, W]
    
    # Maximum frequency is sqrt(0.5^2 + 0.5^2) = 0.707 at corners
    # But let's normalize to the maximum radial frequency in the grid
    r_max = r_freq.max()
    r_norm = (r_freq / r_max).clamp(0, 1)  # [H, W] normalized to [0, 1]
    
    # Map normalized radius to MTF profile indices
    indices = (r_norm * (L - 1)).long().clamp(0, L - 1)  # [H, W]
    
    # Build 2D OTF for each batch
    otf_2d = torch.zeros(B, 1, H, W, device=device, dtype=torch.float32)
    for b in range(B):
        # Use MTF values, ensuring they're positive and <= 1
        mtf_values = mtf_1d[b, 0].clamp(min=0, max=1)
        otf_2d[b, 0] = mtf_values[indices]
    
    return otf_2d