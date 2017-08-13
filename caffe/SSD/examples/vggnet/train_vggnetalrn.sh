#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/VGGNet/VGGNet-A-LRN/solver.prototxt \
    --gpu 0 \
    2>&1 | tee models/VGGNet/VGGNet-A-LRN/train_vggnetalrn.log