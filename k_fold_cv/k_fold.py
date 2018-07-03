# Fine tune VGG16 using k-fold cross validation
# First tunes top dense layers, then 
# top layers with convolutional layers all together

# NOTE: for now, just try with tiny_cnn, compare with old results

from k_fold_dataset.k_fold_dataset import KFoldDataset
import load_models
from sklearn.model_selection import StratifiedKFold
from keras import backend as K

import numpy as np

# global training params
EPOCHS = 10
BATCH_SIZE = 16
IMG_SIZE = (256,256)
COLOR_MODE = 'grayscale'
# 1 channel for grayscale
INPUT_SHAPE = IMG_SIZE + (1,)

# train and evaluate model
def train_fold(dataset, X, Y, train_index, test_index):
  # load mnist model
  model = load_models.mnist(INPUT_SHAPE, dataset.nb_classes)

  # train
  model.fit(X[train_index], Y[train_index],
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = (X[test_index], Y[test_index])
  )

  # evaluate
  score = model.evaluate(X[test_index], Y[test_index])

  # only return accuracy
  return score[1]


# implement k-fold cross validation
def k_fold_cross_val(dataset):
  # load data
  X,Y,class_indices = dataset.get_data(IMG_SIZE, COLOR_MODE)

  # init 5-fold cross validation
  kfold = StratifiedKFold(n_splits=5, shuffle=True)

  # get class labels from one-hot encodings (required for sklearn kfold method)
  labels = [np.argmax(y) for y in Y]


  scores = []
  # train a model for each fold
  for train_index, test_index in kfold.split(X,labels):
    #  top_model = top_model_k_fold
    score = train_fold(dataset,X, Y, train_index, test_index)
    scores.append(score)

    # clear memory
    K.clear_session()
    #  del model
    #  gc.collect()

  print(scores)
  print(np.mean(scores))



def main():
  # CHOOSE DATASET HERE
  dataset_str = 'toronto_line_drawings'

  # init dataset object
  dataset = KFoldDataset(dataset_str)

  k_fold_cross_val(dataset)

main()
