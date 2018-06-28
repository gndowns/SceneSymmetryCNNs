# Agnostic testing script for generating prediction labels given model and dataset
# takes dataset_str as single command line arg

import sys
from dataset.dataset import Dataset

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


# Global train/test params
BATCH_SIZE=16

def generate_predictions(dataset_str, model_file):
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
  y_pred = np.ndarray(shape=(dataset.nb_test_samples))
  for i,p in enumerate(probs):
    y_pred[i] = np.argmax(p)


  return (y_test, y_pred, class_indices)

