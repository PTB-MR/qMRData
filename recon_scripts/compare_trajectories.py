# %%
import numpy as np
import torch
from pathlib import Path
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.data import KData, IData, CsmData
from mrpro.data.traj_calculators import KTrajectoryCartesian
from mrpro.operators import CartesianSamplingOp
import matplotlib.pyplot as plt

# %%
pmain = Path('/example_data')
fnames = [pmain / 'R1_R2/noise_corr_off/9033/IR_T1w_R1.h5', 
         pmain / 'R1_R2/noise_corr_off/9033/IR_T1w_R2.h5',
         pmain / 'LLR/noise_corr_off/9003/IR_T1w_TSE_PF.h5',
         pmain / 'LLR/noise_corr_off/9003/IR_T1w_TSE_underS_R1p7.h5',]

trajectories = ['Fully sampled', 'Poisson \n undersampling R=2', 'Partial Fourier', 'Poisson \n undersampling R=1.7']
fig, ax = plt.subplots(2,4, figsize=(16, 6),  gridspec_kw={'height_ratios': [4, 1]})
for cax in ax.flatten():
    cax.set_xticks([])
    cax.set_yticks([])
for idx, (fname, trajectory) in enumerate(zip(fnames, trajectories, strict=True)):
    kdata = KData.from_file(fname, trajectory=KTrajectoryCartesian())
    recon = DirectReconstruction(kdata)
    idata = recon(kdata)
    
    img = idata.data.abs().squeeze()[2*idata.shape[-3]//3]
    img = img / img.max()
    ax[0, idx].imshow(np.rot90(img), vmax=0.8, cmap='gray')
    ax[0, idx].set_title(trajectory, fontsize=16)
    
    cart_sampling = CartesianSamplingOp(encoding_matrix=kdata.header.encoding_matrix, traj=kdata.traj)
    kdata_sorted = cart_sampling.adjoint(kdata.data)[0]
    kdata_sorted[kdata_sorted.abs() > 0] = 1
    ax[1, idx].imshow(kdata_sorted.abs().squeeze()[...,kdata_sorted.shape[-1]//2], vmax=1.0, cmap='gray')
    ax[1, idx].set_xlabel('ky', fontsize=16)
    ax[1, idx].set_ylabel('kz', fontsize=16)
    
plt.tight_layout()
fig.savefig('/output/compare_trajectories.png', dpi=300)