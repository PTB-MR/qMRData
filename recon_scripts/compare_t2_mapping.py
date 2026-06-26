# %%
import numpy as np
import torch
from pathlib import Path
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.data import KData, IData, CsmData
from mrpro.data.traj_calculators import KTrajectoryCartesian
from mrpro.operators import DictionaryMatchOp
from mrpro.operators.models import MonoExponentialDecay
from cmap import Colormap

# %%
import matplotlib.pyplot as plt
from einops import rearrange
import nibabel as nib
from typing import Literal

from nibabel.orientations import (
    io_orientation,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
)


def show_image(
    qmaps: IData, cmap, vmax: float, rsa_or_spr: Literal["rsa", "spr"] = "rsa", flag_show_roi: bool = False,
) -> None:
    """Show the qualitative images."""
    fig, ax = plt.subplots(len(qmaps), 3, figsize=(12, 8))

    for cax in ax.flatten():
        cax.set_xticks([])
        cax.set_yticks([])

    def orient_images(idata: IData) -> np.array:
        orientation = idata.header.orientation.as_matrix().squeeze()
        if orientation.ndim == 3:
            orientation = orientation.mean(0)
        affine_zyx = torch.cat(
            [
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.cat([torch.zeros((3, 1)), orientation], 1),
            ],
            0,
        )

        data = rearrange(idata.data, "... other z y x-> x y z 1 other (...)")
        img_nii = nib.Nifti1Image(
            data.squeeze().abs().numpy(force=True),
            affine_zyx.flip([0, 1]).numpy(),
            dtype=np.float32,
        )
        # Target orientation (RAS)
        ras_ornt = axcodes2ornt(tuple((rsa_or_spr.upper())))
        transform = ornt_transform(io_orientation(img_nii.affine), ras_ornt)
        ras_data = apply_orientation(img_nii.get_fdata(), transform)
        return ras_data

    def plot_multi_slice_image(qmaps, colorbar_label, cmap, vmax):
        """Plot three slices of M2D image."""
        for idx, qmap in enumerate(qmaps):
            img = orient_images(qmap)
            for slice_idx in range(3):
                im = ax[idx, slice_idx].imshow(
                    np.squeeze(img[slice_idx, :, :]),
                    cmap=cmap,
                    vmin=0,
                    vmax=vmax,
                    origin="lower",
                )
                
            points = [[85,18], [59, 73], [52, 40], [80, 108]]
            h, w = img[slice_idx, :, :].shape
            yy, xx = np.ogrid[:h, :w]

            rad = 5
            for (px, py) in points:
                mask = (xx - px)**2 + (yy - py)**2 <= rad**2
                
                if flag_show_roi:
                    masked = np.ma.masked_where(mask == 0, mask)
                    ax[idx, 1].imshow(masked, cmap='Reds', alpha=0.4, origin='lower', vmin=0, vmax=1)
                    
                mean = img[1][mask].mean()
                std  = img[1][mask].std()

                print(f"Mean: {mean:.4f}, Std: {std:.4f}")
                
                if flag_show_roi:
                    # Find center of mask to place text next to it
                    y_center, x_center = np.argwhere(mask).mean(axis=0)

                    ax[idx, 1].text(
                        x_center + rad + 5,  # slightly to the right of the circle
                        y_center,
                        f"μ={mean:.4f}\nσ={std:.4f}",
                        color='white',
                        fontsize=8,
                        va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5)
                    )
                
            
                
        # Add labels
        ylabel = ['MRpro', 'Vendor DICOM']
        for idx in range(len(ylabel)):
            ax[idx, 0].set_ylabel(ylabel[idx], fontsize=16, labelpad=10)
        
        # Single horizontal colorbar below the last row
        if im is not None:
            # Get bounding box of bottom-left and bottom-right axes
            pos_left  = ax[-1, 0].get_position()
            pos_right = ax[-1, -1].get_position()

            # Create a new axes spanning the full width below the grid
            cbar_ax = fig.add_axes([
                pos_left.x0,              # left edge
                pos_left.y0 - 0.15,       # below the last row (adjust 0.08 as needed)
                pos_right.x1 - pos_left.x0,  # full width of the grid
                0.02                       # height of the colorbar
            ])

            cbar = fig.colorbar(
                im,
                cax=cbar_ax,
                orientation="horizontal",
                label=colorbar_label,
            )
            cbar.ax.tick_params(labelsize=16)        # tick labels
            cbar.set_label(colorbar_label, size=16)  # colorbar label

    plot_multi_slice_image(qmaps, "T2 (s)", cmap, vmax)

    plt.show()
    return fig

# %%
fname = Path('/example_data/Subject12/raw/t2map_trufi_Right.mrd')
image_folder = Path('/example_data/Subject12/dcm/Right_Knee/T2Map_TrueFisp/T2Map_MOCO')

rsa_or_spr = "rsa"  # 'rsa' for knee imaging, 'spr' for brain imaging
t2_prep = [0., 0.025, 0.055, 0., 0.025, 0.06]

# %% Reconstruct images
kdata = KData.from_file(fname, trajectory=KTrajectoryCartesian())
kdata.data *= 100

# Separate slices and contrasts
positions = torch.stack(
    torch.broadcast_tensors(*kdata.header.acq_info.position.zyx), -1
).squeeze()
orientation = torch.stack(
    torch.broadcast_tensors(
        *kdata.header.acq_info.orientation.as_directions()[0].zyx
    )
)
orientation = orientation.squeeze()
if orientation.ndim == 2:
    orientation = orientation[..., 0]
sort_idx = torch.argsort((positions @ orientation).squeeze(), stable=True)

kdata = kdata[sort_idx]
n_slices = kdata.header.acq_info.idx.slice.unique().numel()
kdata = kdata.rearrange("(slice contrast) ... -> contrast slice ...", slice=n_slices)
# We have to calculate the coil maps from one of the contrasts which ideally has high signal for all tissue types
csm = CsmData.from_kdata_inati(kdata[0])

recon = DirectReconstruction(kdata, csm=csm)
idata = recon(kdata)

# Parameter estimation
dictionary = DictionaryMatchOp(MonoExponentialDecay(decay_time=torch.as_tensor(t2_prep, dtype=torch.float32)), index_of_scaling_parameter=0)
dictionary.append(torch.tensor(1.0), torch.linspace(0.01, 0.8, 1000)[None, :])
m0_match, t_match = dictionary(idata.data)

#t_match[idata.data.abs().mean(dim=0) < torch.quantile(idata.data.abs().mean(dim=0), 0.95)*0.25] = 0

recon = DirectReconstruction(kdata[0])
map = recon(kdata[0])
map.data = t_match


# %%
map_dicom = IData.from_dicom_folder(image_folder)

# Sort dicom
positions = torch.stack(
    torch.broadcast_tensors(*map_dicom.header.position.zyx), -1
).squeeze()
orientation = torch.stack(
    torch.broadcast_tensors(
        *map_dicom.header.orientation.as_directions()[0].zyx
    )
)
orientation = orientation.squeeze()
if orientation.ndim == 2:
    orientation = orientation[..., 0]
sort_idx = torch.argsort((positions @ orientation).squeeze(), stable=True)
map_dicom = map_dicom[sort_idx]

# Dicom saves quantitative values in ms, we use seconds here.
map_dicom.data *= 1e-3

vmax = 0.2
fig = show_image([map, map_dicom], Colormap("navia").to_mpl(), vmax, rsa_or_spr, flag_show_roi=False)
fig.savefig('/output/compare_t2_mapping.png', dpi=300)

fig = show_image([map, map_dicom], Colormap("navia").to_mpl(), vmax, rsa_or_spr, flag_show_roi=True)
fig.savefig('/output/compare_t2_mapping_rois.png', dpi=300)



# %%
