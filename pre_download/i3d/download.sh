#!/usr/bin/env bash

python pre_download/i3d/convert_tf_pretrained.py 400

# It will automatically downloads the i3D model pretrained on Kinetics-400 and converts this model PyTorch, resulting in `i3d_pretrained_400.pt`.