#*-*coding:utf-8*-*
import struct
import numpy as np
import  matplotlib.pyplot as plt
import Image

filename='mnist_data/t10k-images-idx3-ubyte'
binfile=open(filename,'rb')
buf=binfile.read()

# read 4 unsigned int32
index=0
magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',buf,index)
index+=struct.calcsize('>IIII')

#将每张图片按照格式存储到对应位置
for image in range(0,numImages):
	im=struct.unpack_from('>784B',buf,index)
	index+=struct.calcsize('>784B')
	
	#这里注意 Image对象的dtype是uint8，需要转换
	im=np.array(im,dtype='uint8')
	im=im.reshape(28,28)
	# fig=plt.figure()
	# plotwindow=fig.add_subplot(111)
	# plt.imshow(im,cmap='gray')
	# plt.show()
	im=Image.fromarray(im)
	im.save('mnist_data/image/test_%s.bmp'%image,'bmp')