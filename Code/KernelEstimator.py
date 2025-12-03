import torch.nn as nn
import torch
class KernelEstimator(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv1d(1,32,3, padding=1),   # [1,1,363] to [1,32,363]
        nn.BatchNorm1d(32),
        nn.LeakyReLU(),
        nn.Conv1d(32,64,3,padding=1),    #[1,32,363] to [1,64,363]
        nn.BatchNorm1d(64),
        nn.LeakyReLU(),
        nn.Conv1d(64,128,3,padding=1),   #[1,64,363] to [1,128,363]
        nn.BatchNorm1d(128),
        nn.LeakyReLU(),
        nn.Conv1d(128,64,3,padding=1),   #[1,128,363] to [1,64,363]
        nn.BatchNorm1d(64),
        nn.LeakyReLU(),
        nn.Conv1d(64,32,3,padding=1),    #[1,64,363] to [1,32,363]
        nn.BatchNorm1d(32),
        nn.LeakyReLU(),
        nn.Conv1d(32,1,3,padding=1),     #[1,32,363] to [1,16,363]
        nn.Sigmoid()
    )

  def forward(self, psd):
    x = self.model(psd)
    #output shape is [1,1,363]
    return x