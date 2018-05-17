# Test Performance of VGG16 Fine Tuned to CIFAR10 Dataset
# Only Run AFTER running `tune.py` to generate fine-tuned VGG model

import sys
import os.path
import numpy

from keras.models import load_model

# for importing cifar10 test data
from load_cifar10 import load_cifar10_data

# Path to tuned vgg model
_MODEL_PATH = './vgg_cifar10_tuned.h5'

def main():
  # load tuned model
  model = load_model(_MODEL_PATH)
  
  # load data
  X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows=224, img_cols=224)
  batch_size = 16

  # Make predictions on test data
  predictions_valid = model.predict(
    X_valid, batch_size=batch_size, verbose=1
  )

  # calculate percent correct
  num_correct = 0
  for p,y in zip(predictions_valid,Y_valid):
    # index of ground truth category
    i = numpy.where(y == max(y))[0][0]
    # index of most likely prediction
    j = numpy.where(p == max(p))[0][0]
    if i==j:
      num_correct = num_correct + 1

  percent_correct = float(num_correct) / float(len(Y_valid))
  print('Num Correct Predictions: %d' %(num_correct))
  print('Out of: %d' %len(Y_valid))
  print('Percent Correct: %.2f' %percent_correct)

# ===============================================

# check if fine-tuned VGG model exists
if not os.path.exists(_MODEL_PATH):
  print('Error: Please run `tune.py` to generate the tuned VGG model')
  sys.exit(1)

# Else if exists, test
main()  

