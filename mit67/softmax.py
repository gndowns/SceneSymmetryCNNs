# Replace softmax and retrain whole network

# NOTE: I attempted the 'steps for reproducibility' mentioned here:
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# however it seems there is still some non-determinism in 
# Keras / TF / GPU operations
# these results are NOT reproducible
# we instead run everything 5 times and take the mean

from mit67_dataset import MIT67Dataset
from vgg16_utils import vgg16_hybrid_1365, vgg16_hybrid_1365_stride, vgg11
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import sys

def train_and_test(datasets, model_str, color_mode):
  train_dataset = datasets[0]

  img_size = (224,224)

  nb_datasets = len(datasets)

  # only load training data for first dataset
  x_train, y_train = train_dataset.train_data(img_size,color_mode,1)

  # load testing data for all datasets
  x_test, y_test = [None]*nb_datasets, [None]*nb_datasets

  for j,d in enumerate(datasets):
    x_test[j], y_test[j] = d.test_data(img_size, color_mode, 1)
  
  # repeat 5 times and take the mean
  # since results are not perfectly reproducible
  scores = [ [None]*5 for _ in range(nb_datasets) ]
  for i in range(5):
    print('trial ' + str(i+1) + ' of 5')

    trial(x_train, y_train, x_test, y_test, scores, model_str, i)

    # clear memory
    K.clear_session()

  # print mean results
  for j,d in enumerate(datasets):
    print(d.str)
    print(scores[j])
    print(np.mean(scores[j]))

  print('done')


# Runs a single train and test trial
def trial(x_train, y_train, x_test, y_test, scores, model_str, i):
  # load proper model 
  if model_str == 'vgg16_hybrid_1365':
    model = vgg16_hybrid_1365(1)
  elif model_str == 'vgg16_hybrid_1365_stride':
    model = vgg16_hybrid_1365_stride(1)
  # places line drawing network
  elif model_str == 'places365_vgg11_runaway_weights.h5':
    model = vgg11(365,1)
    weights_file = 'models/' + model_str
    model.load_weights(weights_file)
    # remove old softmax
    model.pop()
  else:
    print 'ERROR: model not implemented'
    sys.exit()

  # append new softmax layer
  model.add(Dense(y_test[0].shape[1], activation='softmax'))

  # fine tuning stats
  model.compile(
    optimizer=SGD(lr=1e-3,decay=1e-6,momentum=0.9,nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  model.fit(x_train, y_train,
    batch_size = 32,
    # converges very quickly, but need more (10) for mit67
    #  epochs = 5,
    epochs = 10,
    validation_data = (x_test[0], y_test[0])
  )

  # take accuracy score for each dataset
  print('evaluating on all datasets...')
  for j in range(len(x_test)):
    scores[j][i] = model.evaluate(x_test[j], y_test[j])[1]

  return


def main():
  dataset_strs = ['smooth']

  datasets = [MIT67Dataset(s) for s in dataset_strs]

  # choose which vgg16 model to use
  model_str = 'vgg16_hybrid_1365'
  #  model_str = 'vgg16_hybrid_1365_stride'
  #  model_str = 'places365_vgg11_runaway_weights.h5'

  # changes with which model we're using (number of channels)
  #  color_mode = 'grayscale'
  color_mode = 'rgb'

  train_and_test(datasets, model_str, color_mode)

main()
