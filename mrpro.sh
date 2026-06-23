#!/usr/bin/env bash
set -ev

source /opt/conda/bin/activate reco_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install mrpro