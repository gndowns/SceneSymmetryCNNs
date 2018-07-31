# Train an svm on three-channel combined images

from mit67_dataset import MIT67Dataset
from vgg16_utils import vgg16_sequential
from vgg16_hybrid_places_1365 import VGG16_Hybrid_1365
from thundersvmScikit import *
import numpy as np

def train_and_test():
  img_size = (224,224)

  # load intact
  intact_dataset = MIT67Dataset('smooth')
  x_intact_train, y_intact_train = intact_dataset.train_data(img_size, 'grayscale', 1)
  x_intact_test, y_intact_test = intact_dataset.test_data(img_size, 'grayscale', 1)

  sym_dataset = MIT67Dataset('smooth_dR_symmetric')
  x_sym_train, y_sym_train = sym_dataset.train_data(img_size, 'grayscale', 1)
  x_sym_test, y_sym_test = sym_dataset.test_data(img_size, 'grayscale', 1)

  asym_dataset = MIT67Dataset('smooth_dR_asymmetric')
  x_asym_train, y_asym_train = asym_dataset.train_data(img_size, 'grayscale', 1)
  x_asym_test, y_asym_test = asym_dataset.test_data(img_size, 'grayscale', 1)


  # combine datasets into single 3-channel set
  x_train = np.ndarray(shape=(x_intact_train.shape[0:3] + (3,)))
  # put each dataset in resp. channels
  x_train[:,:,:,0] = x_intact_train.squeeze()
  x_train[:,:,:,1] = x_sym_train.squeeze()
  x_train[:,:,:,2] = x_asym_train.squeeze()

  x_test = np.ndarray(shape=(x_intact_test.shape[0:3] + (3,)))
  x_test[:,:,:,0] = x_intact_test.squeeze()
  x_test[:,:,:,1] = x_sym_test.squeeze()
  x_test[:,:,:,2] = x_asym_test.squeeze()

  # convert one hot to labels
  train_labels = [np.argmax(y) for y in y_intact_train]
  test_labels = [np.argmax(y) for y in y_intact_test]

  # load model
  hybrid = VGG16_Hybrid_1365()
  model = vgg16_sequential(1365)
  model.set_weights(hybrid.get_weights())
  model.pop()

  # generate bottleneck features
  bneck_train = model.predict(x_train)
  bneck_test = model.predict(x_test)

  # train a linear svm
  svc = SVC(kernel='linear')
  print('training...')
  svc.fit(bneck_train, train_labels)
  print(m.score(bneck_train, train_labels))
  print(m.score(bneck_test, test_labels))

train_and_test()
