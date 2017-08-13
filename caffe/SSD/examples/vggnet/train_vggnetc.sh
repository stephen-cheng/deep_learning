#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/VGGNet/VGGNet-C/solver.prototxt \
    --gpu 0 \
    2>&1 | tee models/VGGNet/VGGNet-C/train_vggnetc.log