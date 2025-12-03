import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

img = nib.load(r"D:\Charan work file\KernelEstimator\Data_Root\trainA\0B14X41758_filter_B.nii")
data = img.get_fdata()
slice_idx = 30

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
slice_img = ax.imshow(data[:, :, slice_idx], cmap='gray')
ax.set_title(f"Slice {slice_idx}")

axslice = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(axslice, 'Slice', 0, data.shape[2]-1, valinit=slice_idx, valstep=1)

def update(val):
    idx = int(slider.val)
    slice_img.set_data(data[:, :, idx])
    ax.set_title(f"Slice {idx}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()