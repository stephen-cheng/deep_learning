#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/VGGNet/VGGNet-B/solver.prototxt \
    --gpu 0 \
    2>&1 | tee models/VGGNet/VGGNet-B/train_vggnetb.log