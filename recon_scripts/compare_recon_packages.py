# %%
import matplotlib.pyplot as plt
import subprocess as spr
from pathlib import Path
import numpy as np
import h5py

import sigpy.mri as mr

from einops import rearrange

from mrpro.algorithms.reconstruction import IterativeSENSEReconstruction
from mrpro.data.traj_calculators import KTrajectoryCartesian
from mrpro.data import KData, CsmData
from bart import bart



def kdata_ktraj_to_bart(kdata):
    kdat = rearrange(kdata.data, '1 coils z y x -> 1 x (y z) coils')
    ktraj = rearrange(kdata.traj.as_tensor(), 'dim 1 1 z y x -> dim x (y z)')
    return kdat.numpy(), ktraj.numpy()[::-1,...]
    

# %%
fnames = ['/example_data/Subject 10/Qualitative protocol/Subject10-Brain/ISMRMRD/Subject10-Brain-PD.h5',
          '/example_data/Subject 10/Qualitative protocol/Subject10-Brain/ISMRMRD/Subject10-Brain-T1.h5',
          '/example_data/Subject 10/Qualitative protocol/Subject10-Brain/ISMRMRD/Subject10-Brain-T2.h5',]
slice_idx = 1
recons = []
for fname in fnames:
    # PREPARE DATA - MRPRO
    kdata = KData.from_file(fname, KTrajectoryCartesian())
    kdata = kdata.remove_readout_os()
    kdata.data = kdata.data*100

    kdata = kdata[slice_idx] # Select single slice
    kdata.header.recon_matrix.x = kdata.header.encoding_matrix.x
    kdata.header.recon_matrix.y = kdata.header.encoding_matrix.y
    csm = CsmData.from_kdata_inati(kdata)

    # PREPARE DATA - BART
    kdat_bart, ktraj_bart = kdata_ktraj_to_bart(kdata)
    csm_bart = rearrange(csm.data, '1 coils z y x -> x y z coils').numpy()

    # PREPARE DATA - SIGPY
    kdat_sigpy = rearrange(kdat_bart, '1 x yz coils -> coils (x yz)')
    csm_sigpy = rearrange(csm_bart, 'x y 1 coils -> coils x y')
    ktraj_sigpy = rearrange(np.copy(ktraj_bart[:2,...]), 'dim x yz-> (x yz) dim')

    # PREPARE DATA - MRIRECO
    kdat_mrireco = rearrange(kdat_bart, '1 x yz coils -> coils (x yz)')
    csm_mrireco = rearrange(csm_bart, 'x y z coils -> coils z y x')
    ktraj_mrireco = rearrange(np.copy(ktraj_bart[:2,...]), 'dim x yz -> (x yz) dim')
    ktraj_mrireco[...,0] = ktraj_mrireco[...,0]/kdata.header.encoding_matrix.x
    ktraj_mrireco[...,1] = ktraj_mrireco[...,1]/kdata.header.encoding_matrix.y

    with h5py.File('mri_reco_data.h5', 'w') as f:
        f.create_dataset('kdat', data=kdat_mrireco)
        f.create_dataset('ktraj',   data=ktraj_mrireco)
        f.create_dataset('csm',   data=csm_mrireco)
        f.create_dataset('n_k0', data=kdata.shape[-1])
        f.create_dataset('n_k1', data=kdata.shape[-2])
        f.create_dataset('n_x0', data=kdata.header.recon_matrix.x)
        f.create_dataset('n_x1', data=kdata.header.recon_matrix.y)


    # RECON DATA - MRPRO
    recon = IterativeSENSEReconstruction(kdata, csm=csm, n_iterations=1)
    idata = recon(kdata)
    img_mrpro = idata.data.squeeze().abs().numpy()
    img_mrpro /= np.max(img_mrpro)


    #  RECON DATA - MRIRECO_JL
    mrireco_str = f'julia /recon_scripts/compare_mrireco_julia.jl "{fname}"'
    status = spr.run(mrireco_str , shell=True)
    assert status.returncode == 0, 'MriReco.jl reconstruction failed'
    img_mrireco = np.abs(np.squeeze(np.load('mri_reco_output.npz')['img']))
    img_mrireco = rearrange(img_mrireco, 'x y -> y x')
    img_mrireco /= np.max(img_mrireco)


    #  RECON DATA - BART
    img_bart = bart(1, f'pics -r0 -i1 -t', ktraj_bart, kdat_bart, csm_bart)
    img_bart = np.squeeze(np.abs(img_bart))
    img_bart = rearrange(img_bart, 'x y -> y x')
    img_bart /= np.max(img_bart)


    #  RECON DATA - SIGPY
    recon = mr.app.SenseRecon(
        kdat_sigpy,
        csm_sigpy,
        lamda=0.0,
        coord=ktraj_sigpy,
        max_iter=1,
    )
    img_sigpy = recon.run()
    img_sigpy = np.squeeze(np.abs(img_sigpy))
    img_sigpy = rearrange(img_sigpy, 'x y -> y x')
    img_sigpy /= np.max(img_sigpy)


    recons.append([img_bart, img_mrpro, img_mrireco, img_sigpy])


# VIsualisation
fig, ax = plt.subplots(3,4, figsize=(16, 10))
for cidx, (contrast, contrast_name) in enumerate(zip(recons, ['PDw', 'T1w', 'T2w'])):
    for ridx, (img, package_name) in enumerate(zip(contrast, ['BART', 'MRpro', 'MriReco', 'SigPy'])):

        vmin = np.percentile(img, 1)
        vmax = np.percentile(img, 99)
        
        ax[cidx, ridx].imshow(np.rot90(img, -1), cmap='gray', vmin=vmin, vmax=vmax)
            
        # Top: method name (first row only)
        if cidx == 0:
            ax[cidx, ridx].set_title(package_name, fontsize=16)
        
        # Left: raw file name (first column only)
        if ridx == 0:
            ax[cidx, ridx].set_ylabel(contrast_name, fontsize=16)
        
        ax[cidx, ridx].set_xticks([])
        ax[cidx, ridx].set_yticks([])

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/output/reco_packages.png', dpi=300, bbox_inches='tight')
plt.show()

        