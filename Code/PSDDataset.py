import os
import nibabel as nib
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import radial_average  # make sure this function exists

class PSDDataset(Dataset):
    def __init__(self, root_dir):
        self.smooth_dir = os.path.join(root_dir, "trainA")
        self.sharp_dir = os.path.join(root_dir, "trainB")

        smooth_files = os.listdir(self.smooth_dir)
        sharp_files = os.listdir(self.sharp_dir)

        self.pairs = []
        for sfile in smooth_files:
            base_id = sfile.split("_filter_")[0]
            for shfile in sharp_files:
                if base_id in shfile:
                    self.pairs.append((sfile, shfile))
                    break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sfile, shfile = self.pairs[idx]
        s_path = os.path.join(self.smooth_dir, sfile)
        sh_path = os.path.join(self.sharp_dir, shfile)

        I_smooth = nib.load(s_path).get_fdata()
        I_sharp = nib.load(sh_path).get_fdata()

        # normalize intensities
        I_smooth = (I_smooth - I_smooth.min()) / (I_smooth.max() - I_smooth.min() + 1e-8)
        I_sharp = (I_sharp - I_sharp.min()) / (I_sharp.max() - I_sharp.min() + 1e-8)

        # random slice
        z_idx = np.random.randint(0, I_smooth.shape[2])
        I_smooth_slice = I_smooth[:, :, z_idx]
        I_sharp_slice = I_sharp[:, :, z_idx]

        # compute PSDs
        psd_smooth = radial_average(I_smooth_slice)
        psd_sharp = radial_average(I_sharp_slice)

        # convert to tensors
        I_smooth_slice = torch.tensor(I_smooth_slice, dtype=torch.float32).unsqueeze(0)
        I_sharp_slice = torch.tensor(I_sharp_slice, dtype=torch.float32).unsqueeze(0)
        psd_smooth = torch.tensor(psd_smooth, dtype=torch.float32).unsqueeze(0)
        psd_sharp = torch.tensor(psd_sharp, dtype=torch.float32).unsqueeze(0)

        return I_smooth_slice, I_sharp_slice, psd_smooth,psd_sharp

