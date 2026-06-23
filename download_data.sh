#!/usr/bin/env bash
set -ev

# download raw data
wget "https://zenodo.org/records/20516472/files/Subject12.zip?download=1" -O /example_data/Subject12.zip
unzip /example_data/Subject12.zip -d /example_data/