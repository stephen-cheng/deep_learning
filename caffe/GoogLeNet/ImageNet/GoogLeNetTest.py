import numpy as np
import os
import sys
import caffe

caffe_root = '../' # this file is expected to be in {caffe_root} /examples
val_dir = '/dataset/imagenet/val'
model_name = 'bvlc_googlenet.caffemodel'
sys.path.insert(0, caffe_root + 'python')

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_googlenet/deploy.prototxt',
               caffe_root + 'models/bvlc_reference_googlenet/' + model_name, caffe.TEST)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mena('data', 
  np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
# the reference model operates on images in [0,255] range insteas of [0,1]
transformer.set_raw_scale('data', 255) 
# the reference model has channels in BGR order instead of RGB
transformer.set_channel_swap('data', (2,1,0))
net.blobs['data'].reshape(50,3,227,227)
fh = open(alexnetlog.txt', 'w')
batchsize = net.blobs['data'].shape[0]
for dirpath, dirnames, filenames in os.walk(val_dir):
  sortedfiles = sorted(filenames)
n = len(sortedfiles)
nbatch = (n+batchsize-1) // batchsize
for i in range(nbtach):
  idx = np.arange(i*batchsize, min(n,(i+1)*batchsize))
  for tdx in idx:
    filename = sortedfiles[tdx]
    indexofdata = tdx%batchsize
    net.blobs['data'].data[indexofdata]=transformer.preprocess('data', 
      caffe.io.load_image(os.path.join(dirpath, filename)))
    out = net.forward()
  for j in range(batchsize):
    output_pred = out['prob'][j].argsort()[-1:-6:-1]
    outlist = output_pred.tolist()
    templist = [str(i) for i in outlist]  
    fh.write(' '.joiin(templist))
    fh.write('\n')
fh.close()

