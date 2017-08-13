import os, sys
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import h5py

rootDir = './examples/kaggle/facial-keypoints-detection/'

train_csv = rootDir + 'data/training.csv'


train_hd5_name = rootDir + 'data/train.hd5'
val_hd5_name   = rootDir + 'data/val.hd5'

def csv_to_hd5():
    dataframe = read_csv(os.path.expanduser(train_csv))
    dataframe['Image'] = dataframe['Image'].apply( lambda img : np.fromstring(img, sep=' ') )
    dataframe = dataframe.dropna() # remove info lost
    data = np.vstack( dataframe['Image'].values ) / 255.0

    label = dataframe[dataframe.columns[:-1]].values
    label = (label - 48) / 48
    data, label = shuffle( data, label, random_state = 0 ) # random shuffle

    return data, label

if __name__ == '__main__':
    # train data & val data
    data, label = csv_to_hd5()
    data = data.reshape( -1, 1, 96, 96 )
    print type(data)
    print label.max()
    print data.max()
    print label.min()
    print data.min()
    # sys.exit(0)
    data_train = data[:-100, :, :, :]
    data_val = data[-100:, :, :, :]  # last 100 pics as validation
    #train label & val label
    label = label.reshape( -1, 1, 1, 30 )
    label_train = label[:-100, :, :, :]
    label_val   = label[-100:, :, :, :]

    fhandle = h5py.File( train_hd5_name, 'w' ) # train dataset
    fhandle.create_dataset( 'data', data = data_train, compression = 'gzip', compression_opts = 4 )
    fhandle.create_dataset( 'label', data = label_train, compression = 'gzip', compression_opts = 4 )
    fhandle.close()

    fhandle = h5py.File( val_hd5_name, 'w' ) # validation dataset
    fhandle.create_dataset( 'data', data = data_val, compression = 'gzip', compression_opts = 4 )
    fhandle.create_dataset( 'label', data = label_val, compression = 'gzip', compression_opts = 4 )
    fhandle.close()

