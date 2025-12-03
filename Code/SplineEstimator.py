import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline

class FixedSplineLayer(nn.Module):
    def __init__(self, num_output_points=363):
        super().__init__()
        
        knots_np = np.array([
            0., 0., 0., 0., 
            0.00167847, 0.00281727, 0.00472871, 0.00793701, 
            0.01332204, 0.02236068, 0.03753178, 0.06299605, 
            0.10573713, 0.17747683, 0.29788994, 
            0.5, 0.5, 0.5, 0.5
        ])
        
        degree = 3
        num_control_points = len(knots_np) - degree - 1 # Should be 15
        
        x_eval = np.linspace(0.0, 0.5, num_output_points)
        basis_matrix_np = BSpline.design_matrix(x_eval, knots_np, degree).toarray()
        self.register_buffer('basis_matrix', torch.tensor(basis_matrix_np, dtype=torch.float32))

    def forward(self, control_points):
        return torch.matmul(control_points, self.basis_matrix.T)


class KernelEstimator(nn.Module):
  def __init__(self):
    super().__init__()
    self.features = nn.Sequential(
        nn.Conv1d(1, 32, 3, padding=1),    # [1, 32, 363]
        nn.BatchNorm1d(32),
        nn.LeakyReLU(),
        
        nn.Conv1d(32, 64, 3, padding=1),   # [1, 64, 363]
        nn.BatchNorm1d(64),
        nn.LeakyReLU(),
        
        nn.Conv1d(64, 128, 3, padding=1),  # [1, 128, 363]
        nn.BatchNorm1d(128),
        nn.LeakyReLU(),
        
        nn.Conv1d(128, 64, 3, padding=1),  # [1, 64, 363]
        nn.BatchNorm1d(64),
        nn.LeakyReLU(),
        
        nn.Conv1d(64, 32, 3, padding=1),   # [1, 32, 363]
        nn.BatchNorm1d(32),
        nn.LeakyReLU()
    )
    
    self.flatten = nn.Flatten() # [1,32,363] -> [32*363]
    self.fc_head = nn.Linear(32 * 363, 15)  # 15 points
    self.activation = nn.Softplus(beta = 2.0)
    self.spline_layer = FixedSplineLayer(num_output_points=363) 

  def forward(self, psd):
    x = self.features(psd)
    x = self.flatten(x)
    control = self.fc_head(x)
    control = self.activation(control)
    control = control / control.max(dim=1, keepdim=True).values
    mtf_curve = self.spline_layer(control)
    return mtf_curve.unsqueeze(1)