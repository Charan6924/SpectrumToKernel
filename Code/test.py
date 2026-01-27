import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

#load and view nii volume
img = nib.load(r"D:\Charan work file\KernelEstimator\Data_Root\trainB\0B14X41758_filter_E.nii")  #type: ignore
data = img.get_fdata() #type: ignore

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
slice_idx = data.shape[2] // 2
slice_img = ax.imshow(data[:, :, slice_idx], cmap='gray') #type: ignore
ax.set_title(f"Slice {slice_idx}") #type: ignore

axslice = plt.axes([0.25, 0.1, 0.5, 0.03]) #type: ignore
slider = Slider(axslice, 'Slice', 0, data.shape[2]-1, valinit=slice_idx, valstep=1) #type: ignore

def update(val):
    idx = int(slider.val)
    slice_img.set_data(data[:, :, idx])
    ax.set_title(f"Slice {idx}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()