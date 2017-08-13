
./build/tools/caffe train --solver=./examples/kaggle/facial-keypoints-detection/fk_solver.prototxt 2>&1 | tee ./log/fk_train.log
