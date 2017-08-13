load data:				unzip_ILSVRC2012.sh
						unzip.sh
			
LMDB info:				get_ilsvrc_aux.sh

resize img to lmdb:		create_imagenet.sh

subtract pixel mean:	make_imagenet_mean.sh
			
model train:			./build/tools/caffe train --solver=models/bvlc_alexnet/solver.prototxt
						(nohup ./build/tools/caffe train --solver=models/bvlc_alexnet/solver.prototxt &)
						
imagenet test:          AlexNetTest.py
				

