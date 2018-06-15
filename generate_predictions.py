# Agnostic testing script for generating prediction labels given model and dataset
# takes dataset_str as single command line arg

import sys
from dataset.dataset import Dataset

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from scipy.io import savemat

# Global train/test params
BATCH_SIZE=16

def main(dataset_str):
  # h5 file of saved model
  #  model_file = 'toronto_line_drawings_tiny_cnn.h5'
  model_file = 'toronto_line_drawings_top_conv_block.h5'

  # load dataset
  print('loading dataset ' + dataset_str + '...')
  dataset = Dataset(dataset_str)
  # load saved model
  print('loading model...')
  model = load_model(model_file)

  # grayscale or rgb, based on requirements of model
  color_mode = 'rgb' if model.input_shape[3]==3 else 'grayscale'

  # generate test data with labels
  x_test, y_test, class_indices = dataset.test_batch(color_mode)

  print('generating predictions...')
  batch_size = 16
  probs = model.predict(x_test, verbose=1)
  # convert prob vectors to single predictions
  preds = np.ndarray(shape=(dataset.nb_test_samples))
  for i,p in enumerate(probs):
    preds[i] = np.argmax(p)


  # save predictions and labels as matlab arrays
  print('saving predictions...')
  savemat(dataset_str + '_test.mat', {'preds': preds, 'labels': y_test})

  print('class indices: ')
  print(class_indices)

# check for command line argument - dataset_str
if not len(sys.argv) == 2: 
  print('Error: not enough arguments. Must call with name of desired dataset')
  sys.exit()

# call with first arg as dataset_str
main(sys.argv[1])

