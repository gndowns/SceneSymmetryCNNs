# Combine softmax & 3 channel encoding setups

from mit67_dataset import MIT67Dataset
from vgg16_utils import vgg16_hybrid_1365
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K
import numpy as np

def train_and_test(datasets):
  train_dataset = datasets[0]


  img_size = (224,224)
  color_mode = 'grayscale'

  # load each dataset
  x1_train, y1_train = train_dataset.train_data(img_size,color_mode,1)
  x1_test, y1_test = train_dataset.test_data(img_size,color_mode,1)
  x2_train, y2_train = train_dataset.train_data(img_size,color_mode,1)
  x2_test, y2_test = train_dataset.test_data(img_size,color_mode,1)
  x3_train, y3_train = train_dataset.train_data(img_size,color_mode,1)
  x3_test, y3_test = train_dataset.test_data(img_size,color_mode,1)

  x_train = np.ndarray(shape=(x1_train.shape[0:3] + (3,)))
  x_train[:,:,:,0] = x1_train.squeeze()
  x_train[:,:,:,1] = x2_train.squeeze()
  x_train[:,:,:,2] = x3_train.squeeze()

  x_test = np.ndarray(shape=(x1_test.shape[0:3] + (3,)))
  x_test[:,:,:,0] = x1_test.squeeze()
  x_test[:,:,:,1] = x2_test.squeeze()
  x_test[:,:,:,2] = x3_test.squeeze()

  # labels are the same for each
  y_train = y1_train
  y_test = y1_test

  # train 5 times and take the mean
  scores = [None] * 5
  for i in range(5):
    print('trial ' + str(i+1) + ' of 5')

    model = vgg16_hybrid_1365(1)

    # append new softmax layer
    model.add(Dense(train_dataset.nb_classes, activation='softmax'))

    # fine tuning stats
    model.compile(
      optimizer=SGD(lr=1e-3,decay=1e-6,momentum=0.9,nesterov=True),
      loss='categorical_crossentropy',
      metrics=['accuracy']
    )

    model.fit(x_train, y_train,
      batch_size = 32,
      # converges very quickly, but takes more time than intact
      epochs = 10,
      validation_data = (x_test, y_test)
    )

    scores[i] = model.evaluate(x_test,y_test)[1]

    K.clear_session()

  print(scores)
  print(np.mean(scores))
  print('done')



def main():
  # choose 3 datasets
  dataset_strs = ['smooth', 'dR_weighted', 'd2R_weighted']

  datasets = [MIT67Dataset(s) for s in dataset_strs]

  train_and_test(datasets)

main()
