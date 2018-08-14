# Train an SVM on combined output of 
# VGG16 rgb and VGG16 three-channel

from vgg16_utils import vgg16_hybrid_1365
from mit67_dataset import MIT67Dataset
from thundersvmScikit import SVC
import numpy as np

np.random.seed(2018)

# save bottleneck features
# since doing it all at once causes seg fault
def bottleneck_features():
  model = vgg16_hybrid_1365(1)
  
  img_size = (224,224)
  color_mode = 'rgb'

  # Load RGB dataset
  rgb_dataset = MIT67Dataset('rgb')

  x_rgb_train, y_rgb_train = rgb_dataset.train_data(img_size,color_mode,1)
  x_rgb_test, y_rgb_test = rgb_dataset.test_data(img_size,color_mode,1)

  rgb_train_labels = np.asarray([np.argmax(y) for y in y_rgb_train])
  rgb_test_labels = np.asarray([np.argmax(y) for y in y_rgb_test])

  color_mode = 'grayscale'

  # load weighted line drawings
  intact_dataset = MIT67Dataset('smooth')
  x1_train, y1_train = intact_dataset.train_data(img_size,color_mode,1)
  x1_test, y1_test = intact_dataset.test_data(img_size,color_mode,1)

  dR_dataset = MIT67Dataset('dR_weighted')
  x2_train, y2_train = dR_dataset.train_data(img_size,color_mode,1)
  x2_test, y2_test = dR_dataset.test_data(img_size,color_mode,1)

  d2R_dataset = MIT67Dataset('d2R_weighted')
  x3_train, y3_train = d2R_dataset.train_data(img_size,color_mode,1)
  x3_test, y3_test = d2R_dataset.test_data(img_size,color_mode,1)

  x_train = np.ndarray(shape=(x1_train.shape[0:3] + (3,)))
  x_test = np.ndarray(shape=(x1_test.shape[0:3] + (3,)))
  # put each dataset in respective channels
  x_train[:,:,:,0] = x1_train.squeeze()
  x_train[:,:,:,1] = x2_train.squeeze()
  x_train[:,:,:,2] = x3_train.squeeze()
  x_test[:,:,:,0] = x1_test.squeeze()
  x_test[:,:,:,1] = x2_test.squeeze()
  x_test[:,:,:,2] = x3_test.squeeze()

  x_train_labels = np.asarray([np.argmax(y) for y in y1_train])
  x_test_labels = np.asarray([np.argmax(y) for y in y1_test])

  # get rgb bottleneck features
  rgb_bneck_train = model.predict(x_rgb_train)
  rgb_bneck_test = model.predict(x_rgb_test)

  # other bneck features
  x_bneck_train = model.predict(x_train)
  x_bneck_test = model.predict(x_test)

  # concatenate features
  bneck_train = np.concatenate((rgb_bneck_train, x_bneck_train), axis=0)
  bneck_test = np.concatenate((rgb_bneck_test, x_bneck_test), axis=0)

  train_labels = np.concatenate((rgb_train_labels, x_train_labels), axis=0)
  test_labels = np.concatenate((rgb_test_labels, x_test_labels), axis=0)

  np.save('bneck_train', bneck_train)
  np.save('bneck_test', bneck_test)
  np.save('train_labels', train_labels)
  np.save('test_labels', test_labels)

  print('done')

# use saved bneck features
def train_and_test():
  bneck_train = np.load('bneck_train.npy')
  bneck_test = np.load('bneck_test.npy')
  train_labels = np.load('train_labels.npy')
  test_labels = np.load('test_labels.npy')

  svc = SVC(kernel='linear')
  svc.fit(bneck_train, train_labels)
  score = svc.score(bneck_test, test_labels)

  print(score)

  
def main():
  if 0:
    bottleneck_features()
  else:
    train_and_test()

main()
