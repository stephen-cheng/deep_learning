import os
import skimage.io
from skimage import transform


def img_resize(path):
	imglist=os.listdir(path)
	for imgs in imglist:
		path_child = path+'/'+imgs
		for img_name in os.listdir(path_child):
			img=skimage.io.imread(path_child+'/'+img_name)
			dst=transform.resize(img, (128, 128))
			skimage.io.imsave(path_child+'/'+img_name, dst)
		print('%s resized successfully!' % imgs)
	
if __name__=='__main__':
	path_train="dataset/train"
	path_test="dataset/test"
	img_resize(path_train)
	img_resize(path_test)

