from __future__ import print_function

caffe_root = './' # this file is expected to be in {caffe_root}/examples/{net}, and is expected to be run in {caffe_root}
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess

resume_training = True
remove_old_models = False
run_soon = True
draw = True
# plot = False

def ZFNetBody(net, from_layer, for_training=True):
  net.conv1 = L.Convolution(net[from_layer], kernel_size=k_conv1, stride=s_conv1, num_output=d_conv1, pad=p_conv1, 
                            bias_term=True, weight_filler=dict(type='gaussian',std=0.01), bias_filler=dict(type='constant',std=0), 
                            param=[dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)])
  net.relu1 = L.ReLU(net.conv1, in_place=True)
  net.pool1 = L.Pooling(net.relu1, pool=P.Pooling.MAX, kernel_size=k_pool1, stride=s_pool1)
  net.norm1 = L.LRN(net.pool1, lrn_param=dict(local_size=local_size_norm1, alpha=alpha_norm1, beta=beta_norm1))
  
  net.conv2 = L.Convolution(net.norm1, kernel_size=k_conv2, stride=s_conv2, num_output=d_conv2, #pad=p_conv2, 
                            bias_term=True, weight_filler=dict(type='gaussian',std=0.01), bias_filler=dict(type='constant',std=0), 
                            param=[dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)])
  net.relu2 = L.ReLU(net.conv2, in_place=True)
  net.pool2 = L.Pooling(net.relu2, pool=P.Pooling.MAX, kernel_size=k_pool2, stride=s_pool2)
  net.norm2 = L.LRN(net.pool2, lrn_param=dict(local_size=local_size_norm2, alpha=alpha_norm2, beta=beta_norm2))
  
  net.conv3 = L.Convolution(net.norm2, kernel_size=k_conv3, stride=s_conv3, num_output=d_conv3, pad=p_conv3, 
                            bias_term=True, weight_filler=dict(type='gaussian',std=0.01), bias_filler=dict(type='constant',std=0), 
                            param=[dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)])
  net.relu3 = L.ReLU(net.conv3, in_place=True)
  
  net.conv4 = L.Convolution(net.relu3, kernel_size=k_conv4, stride=s_conv4, num_output=d_conv4, pad=p_conv4, 
                            bias_term=True, weight_filler=dict(type='gaussian',std=0.01), bias_filler=dict(type='constant',std=0), 
                            param=[dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)])
  net.relu4 = L.ReLU(net.conv4, in_place=True)
  
  net.conv5 = L.Convolution(net.relu4, kernel_size=k_conv5, stride=s_conv5, num_output=d_conv5, pad=p_conv5, 
                            bias_term=True, weight_filler=dict(type='gaussian',std=0.01), bias_filler=dict(type='constant',std=0), 
                            param=[dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)])
  net.relu5 = L.ReLU(net.conv5, in_place=True)
  net.pool5 = L.Pooling(net.relu5, pool=P.Pooling.MAX, kernel_size=k_pool5, stride=s_pool5)
  
  net.fc6 = L.InnerProduct(net.pool5, num_output=k_ip6,
                           weight_filler=dict(type='gaussian',std=0.01), bias_filler=dict(type='constant',std=0), 
                           param=[dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)])
  net.relu6 = L.ReLU(net.fc6, in_place=True)
  net.drop6 = L.Dropout(net.relu6, dropout_param=dict(dropout_ratio=r_drop6), in_place=True)
  
  net.fc7 = L.InnerProduct(net.fc6, num_output=k_ip7,
                           weight_filler=dict(type='gaussian',std=0.01), bias_filler=dict(type='constant',std=0), 
                           param=[dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)])
  net.relu7 = L.ReLU(net.fc7, in_place=True)
  net.drop7 = L.Dropout(net.relu7, dropout_param=dict(dropout_ratio=r_drop7), in_place=True)
  
  net.fc8 = L.InnerProduct(net.fc7, num_output=k_ip8,
                           weight_filler=dict(type='gaussian',std=0.01), bias_filler=dict(type='constant',std=0), 
                           param=[dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)])
  if not for_training:
    net.acc = L.Accuracy(net.fc8, net.label, include=dict(phase=caffe_pb2.Phase.Value('TEST')))
  
  net.loss = L.SoftmaxWithLoss(net.fc8, net.label)
  
  return net

# Directories and filenames
job_name = "ZFNet"
dataset_name = "ilsvrc12"
model_name = "{}_{}".format(job_name, dataset_name)

# dirs
save_dir = "models/{}/{}".format(job_name, dataset_name)
snapshot_dir = "models/{}/{}".format(job_name, dataset_name)
job_dir = "jobs/{}/{}".format(job_name, dataset_name)

# model definition files
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
job_file = "{}/{}.sh".format(job_dir, model_name)
train_net_graph = "{}/train.png".format(job_dir)
test_net_graph = "{}/test.png".format(job_dir)
deploy_net_graph = "{}/deploy.png".format(job_dir)

# model definition parameters
# data layer
train_data = "examples/ilsvrc12_train_lmdb"
test_data = "examples/ilsvrc12_val_lmdb"
mean_file = "data/ilsvrc12/imagenet_mean.binaryproto"
train_batch_size = 128
test_batch_size = 50
crop_size = 225

# conv1
k_conv1 = 7
p_conv1 = 0
s_conv1 = 2
d_conv1 = 96

# pool1
k_pool1 = 3
s_pool1 = 2

# lrn1
local_size_norm1 = 5
alpha_norm1 = 0.0001
beta_norm1 = 0.75

# conv2
k_conv2 = 5
# p_conv2 = 2
s_conv2 = 2
d_conv2 = 256

# pool2
k_pool2 = 3
s_pool2 = 2

# lrn2
local_size_norm2 = 5
alpha_norm2 = 0.0001
beta_norm2 = 0.75

# conv3
k_conv3 = 3
p_conv3 = 1
s_conv3 = 1
d_conv3 = 384

# conv4
k_conv4 = 3
p_conv4 = 1
s_conv4 = 1
d_conv4 = 384

# conv5
k_conv5 = 3
p_conv5 = 1
s_conv5 = 1
d_conv5 = 256

# pool5
k_pool5 = 3
s_pool5 = 2

# ip6
k_ip6 = 4096
r_drop6 = 0.5

# ip7
k_ip7 = 4096
r_drop7 = 0.5

# ip8
k_ip8 = 1000

# solver parameters
gpus = "0"
solver_param = {
  'base_lr': 0.005,
  'lr_policy': "step",
  'stepsize': 200000,
  'weight_decay': 0.0005,
  'gamma': 0.1,
  'momentum': 0.9,
  'max_iter': 700000,
  'snapshot': 20000,
  'solver_mode': P.Solver.GPU,
  'display': 20,
  'test_iter': [2000],
  'test_interval': 2000,
}

# Check file.
check_if_exist(train_data)
check_if_exist(test_data)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

# Create train net
train_net = caffe.NetSpec()
train_net.data, train_net.label = L.Data(source=train_data, backend=P.Data.LMDB, batch_size=train_batch_size, ntop=2, 
                                         transform_param=dict(crop_size=crop_size, mean_file=mean_file, mirror=True), 
                                         include=dict(phase=caffe_pb2.Phase.Value('TRAIN')))

ZFNetBody(train_net, from_layer='data', for_training=True)

with open(train_net_file, 'w') as f:
  print('name: "{}_train"'.format(model_name), file=f)
  print(train_net.to_proto(), file=f)
shutil.copy(train_net_file, job_dir)

# Create test net
test_net = caffe.NetSpec()
test_net.data, test_net.label = L.Data(source=test_data, backend=P.Data.LMDB, batch_size=test_batch_size, ntop=2, 
                                       transform_param=dict(crop_size=crop_size, mean_file=mean_file, mirror=True), 
                                       include=dict(phase=caffe_pb2.Phase.Value('TEST')))

ZFNetBody(test_net, from_layer='data', for_training=False)

with open(test_net_file, 'w') as f:
  print('name: "{}_test"'.format(model_name), file=f)
  print(test_net.to_proto(), file=f)
shutil.copy(test_net_file, job_dir)

# Create deploy net
deploy_net = train_net

with open(deploy_net_file, 'w') as f:
  net_param = deploy_net.to_proto()
  del net_param.layer[0]
  del net_param.layer[-1]
  net_param.name = '{}_deploy'.format(model_name)  
  net_param.input.extend(['data'])
  net_param.input_shape.extend([caffe_pb2.BlobShape(dim=[1, 3, crop_size, crop_size])])
  print(net_param, file=f)
shutil.copy(deploy_net_file, job_dir)

# Create solver
solver = caffe_pb2.SolverParameter(train_net=train_net_file, test_net=[test_net_file], snapshot_prefix=snapshot_prefix, **solver_param)

with open(solver_file, 'w') as f:
  print(solver, file=f)
shutil.copy(solver_file, job_dir)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  if draw:
    f.write('python ./python/draw_net.py {} {} \n'.format(train_net_file, train_net_graph))
    f.write('python ./python/draw_net.py {} {} \n'.format(test_net_file, test_net_graph))
    f.write('python ./python/draw_net.py {} {} \n'.format(deploy_net_file, deploy_net_graph))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  if max_iter > 0: 
    f.write(train_src_param)
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))
  # if plot:
    # f.write('python ./tools/extra/plot_training_log.py 0 {}/{}_test_acc_vs_iter.png {}/{}.log\n'.format(job_dir, model_name, job_dir, model_name))
    # f.write('python ./tools/extra/plot_training_log.py 1 {}/{}_test_acc_vs_time.png {}/{}.log\n'.format(job_dir, model_name, job_dir, model_name))
    # f.write('python ./tools/extra/plot_training_log.py 2 {}/{}_test_loss_vs_iter.png {}/{}.log\n'.format(job_dir, model_name, job_dir, model_name))
    # f.write('python ./tools/extra/plot_training_log.py 3 {}/{}_test_loss_vs_time.png {}/{}.log\n'.format(job_dir, model_name, job_dir, model_name))
    # f.write('python ./tools/extra/plot_training_log.py 4 {}/{}_lr_acc_vs_iter.png {}/{}.log\n'.format(job_dir, model_name, job_dir, model_name))
    # f.write('python ./tools/extra/plot_training_log.py 5 {}/{}_lr_acc_vs_time.png {}/{}.log\n'.format(job_dir, model_name, job_dir, model_name))
    # f.write('python ./tools/extra/plot_training_log.py 6 {}/{}_train_loss_vs_iter.png {}/{}.log\n'.format(job_dir, model_name, job_dir, model_name))
    # f.write('python ./tools/extra/plot_training_log.py 7 {}/{}_train_loss_vs_time.png {}/{}.log\n'.format(job_dir, model_name, job_dir, model_name))

# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)
