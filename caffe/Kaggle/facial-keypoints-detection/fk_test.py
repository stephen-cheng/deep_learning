import numpy as np
import pandas as pd
import caffe
import sys
import csv
import cv2

def savePredictionAsCsvFile( pred, fn, lookupfn ):
    featureNames = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y' , 'left_eye_inner_corner_x', 
                    'left_eye_inner_corner_y', 'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
                    'right_eye_outer_corner_x', 'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x',
                    'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
                    'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x',
                    'mouth_right_corner_y', 'mouth_center_top_lip_x', 'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']
    
    # lookupReader = csv.reader(open(lookupfn), delimiter=",")
    table = []
    no = 0
    for line in open( lookupfn ):
        if no == 0:
            no = 1
        else:
            a, b, c = line.split(",")
            table.append([int(a), int(b), c[:-2]])

    writer = csv.writer(file(fn, 'wb'))
    # writer.writerow(['RowId', 'ImageId', 'FeatureName', 'Location'])
    writer.writerow(['RowId', 'Location'])
    saveId = 0
    r, c = pred.shape
    print r, c
    for ii in range( r ):
        for jj in range(c):
            imgId = ii+1
            featureName = featureNames[jj]
            # apply saturation
            loc = pred[ii, jj]
            loc = max(0, loc)
            loc = min(96, loc)
            # print imgId, table[saveId][1], ' ==== ', featureName, table[saveId][2]

            if imgId == table[saveId][1] and featureName == table[saveId][2]:
                saveId = saveId + 1
                content = [saveId, loc]
                # content = [saveId, imgId, featureName, loc]
                writer.writerow( content )
                #print saveId  

rootDir = './examples/kaggle/facial-keypoints-detection/'
model_file = rootDir + 'fk_deploy.prototxt'
pretrianed = rootDir + 'fk_iter_10000.caffemodel'

test_csv = rootDir + 'data/test.csv'

saveFn = rootDir + 'data/fk_output.csv'
lookupFn = rootDir + 'data/IdLookupTable.csv'

dataframe = pd.read_csv( test_csv, header = 0 )
dataframe['Image'] = dataframe['Image'].apply( lambda img : np.fromstring(img, sep=' ') )

data = np.vstack( dataframe['Image'].values )
data = data.reshape([-1, 96, 96])

data = data.astype( np.float32 )

# scale between 0~1
data = data / 255.0

data = data.reshape( -1, 1, 96, 96 )
# uncomment it if you want to show the image.
# cv2.imshow("Image", data[10,0,:, :])
# cv2.waitKey(0)
# sys.exit(0)

net = caffe.Net( model_file, caffe.TEST )
caffe.set_mode_gpu()

total_images = data.shape[0]
print 'total images to be predicted: ', total_images

# if gpu memory is not enough, you can use the following..
# batch_size = 1
# iter_num = total_images / batch_size
# allPred = np.zeros((total_images, 30))
# for ii in range(iter_num):
# # for ii in range(5):
#     print ii
#     startIndex = ii*batch_size
#     endIndex = ii*batch_size+batch_size
#     endIndex = min( endIndex, total_images )

#     dataL = np.zeros( [endIndex-startIndex, 1, 1, 1], np.float32 )

#     net.set_input_arrays( data[startIndex:endIndex,:,:,:].astype(np.float32), dataL.astype(np.float32) )
#     pred = net.forward()
#     predicted = net.blobs['ip2'].data * 96
#     allPred[startIndex:endIndex, :] = predicted


dataL = np.zeros( [total_images, 1, 1, 1], np.float32 )

net.set_input_arrays( data.astype(np.float32), dataL.astype(np.float32) )
pred = net.forward()

predicted = net.blobs['ip2'].data * 48 + 48
print predicted.max(), predicted.min()

print 'predicted: ', predicted

print 'predicted shape : ', predicted.shape
print 'saving to csv ...'
res = savePredictionAsCsvFile( predicted, saveFn, lookupFn )
# np.savetxt( saveFn, res, delimiter=',' )



        