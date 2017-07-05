import numpy as np

def load_data(filename, flag):
  f = open(filename, 'r')
  if flag == 'data':
    data_array = [np.array(item, dtype=np.float32) for item in [
                 line.replace('  ', ' ').strip().split(' ') for
                 line in f]]
  else:
    data_array = [np.array(item, dtype=np.int32) for item in [
                 line.replace('  ', ' ').strip().split(' ') for
                 line in f]]    

  f.close()
  return np.array(data_array)

if __name__ == '__main__':
  filename = "../../dataset/UCI HAR Dataset/train/X_train.txt"
  data = load_data(filename, 'data')
  print data
  print data.shape
