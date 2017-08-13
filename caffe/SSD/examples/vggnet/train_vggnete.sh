#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/VGGNet/VGGNet-E/solver.prototxt \
    --gpu 0 \
    2>&1 | tee models/VGGNet/VGGNet-E/train_vggnete.log