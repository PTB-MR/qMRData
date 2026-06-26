#!/usr/bin/env bash
set -ev


# download data for comparison of recon packages
wget "https://zenodo.org/records/18847561/files/Subject 10.zip?download=1" -O "/example_data/Subject 10.zip"
unzip "/example_data/Subject 10.zip" -d /example_data/
rm "/example_data/Subject 10.zip"

# download data for comparison of trajectories
wget "https://zenodo.org/records/19661402/files/LLR.zip?download=1" -O /example_data/LLR.zip
unzip /example_data/LLR.zip -d /example_data/
rm /example_data/LLR.zip

wget "https://zenodo.org/records/19661402/files/R1_R2.zip?download=1" -O /example_data/R1_R2.zip
unzip /example_data/R1_R2.zip -d /example_data/
rm /example_data/R1_R2.zip

# download data for comparison of T2 mapping
wget "https://zenodo.org/records/20516472/files/Subject12.zip?download=1" -O /example_data/Subject12.zip
unzip /example_data/Subject12.zip -d /example_data/
rm /example_data/Subject12.zip