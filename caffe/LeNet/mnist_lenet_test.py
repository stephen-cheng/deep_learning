import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
#caffe_root = '/home/stephen/caffe-master'
#sys.path.insert(0, caffe_root + 'python')
import caffe

#model_file = '/home/stephen/caffe-master/examples/mnist/lenet.prototxt'
model_file = 'lenet.prototxt'
pretrained = 'model/lenet_iter_10000.caffemodel'
image_file = 'mnist_data/image/test_23.bmp'

input_image = caffe.io.load_image(image_file, color=False)
#print input_image
#net = caffe.Classifier(model_file, pretrained, channel_swap=(2,1,0), image_dims=(28,28))
#net = caffe.Classifier(model_file, pretrained, raw_scale=255, image_dims=(28,28))
net = caffe.Classifier(model_file, pretrained)
prediction = net.predict([input_image], oversample=False)
caffe.set_mode_cpu()
print('predicted class: ', prediction[0].argmax())