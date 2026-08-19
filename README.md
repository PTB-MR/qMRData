# Multi-site quantitative MR data

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/mrpro/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Image reconstruction and processing of qualitative and quantitative MR raw data acquired at field strength ranging from 55mT to 0.6T.

The raw k-space data and reconstructed image data can be found at the following repositories:
- [i3M (Valencia, Spain), Physio1](https://zenodo.org/records/18460613)
- [IBT-CMR (Zurich, Switzerland), Philips Ingenia Ambition X, ramped down to 0.6T](https://zenodo.org/records/18847561)
- [PUC (Santiago de Chile, Chile), Siemens MAGNETOM Free.Max 0.55T](https://doi.org/10.5281/zenodo.19189305)
- [Leiden University Medical Center (Leiden, The Netherlands)](https://zenodo.org/records/19661402)

## Reconstruction with different software packages

We demonstrate how the provided data can be reconstructed with these reconstruction packages:
- [BART](https://mrirecon.codeberg.page/)
- [MRpro](https://mrpro.rocks/)
- [SigPy](https://sigpy.readthedocs.io/en/latest/)
- [MriReco](https://github.com/MagneticResonanceImaging/MRIReco.jl)

To install all these packages in the same environment, we provide a Dockerfile. Simply build the Docker image
```
docker build -t qmri_data .
```
start the container with a mapped folder to save the obtained figure in
```
docker run --rm -it -v /output:/output qmri_data
```
activate the environment
```
conda activate reco_env
```
and then run any of the reconstruction scripts
```
python recon_scripts/compare_recon_packages.py 
```
```
python recon_scripts/compare_t2_mapping.py 
```
```
python recon_scripts/compare_trajectories.py 
```


