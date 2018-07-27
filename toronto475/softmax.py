# Replace the softmax layer of VGG16 and re-train the whole network
# Then run an SVM on bottleneck features
# nvmd no SVM neededo
# maybe try svm and compare with plain classification
# TODO: implement 3channel version
# make them use combined train() method for simplicity

# this doesn't seem to be workig well for 3 channel...
# maybe train more fully connected layers from scratch first
# don't kill yourself tuning the learnign rate and what not,
# the trends should be evident either way

# this method 'works' for the most part,
# but it always seems theres ~1 fold that doesn't converge at all
# under this setup, regardless of number of epochs

from k_fold_dataset import KFoldDataset
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from keras.layers import Dense
from keras.optimizers import SGD
from thundersvmScikit import SVC
from vgg16_utils import vgg16_hybrid_1365
import numpy as np

# reproducibility
np.random.seed(2018)

# top-level function
def train_and_test(train_datasets):
  # check if single training dataset or multiple
  if len(train_datasets) == 1:
    train_and_test_single(train_datasets)
  elif len(train_datasets) == 3:
    train_and_test_multi(train_datasets)
  else:
    print('Error: need 1 or 3 training datasets')
    return

def train_and_test_single(train_datasets):
  # only use single training dataset
  train_dataset = train_datasets[0]


  img_size = (224, 224)
  color_mode = 'rgb'

  X,Y = train_dataset.get_data(img_size, color_mode, 1)

  kfold = StratifiedKFold(n_splits=5, shuffle=True)

  labels = np.asarray([np.argmax(y) for y in Y])

  scores = [None] * 5
  i=0
  for train_idx, test_idx in kfold.split(X, labels):
    print('fold ' + str(i+1) + ' of 5')

    # import vgg16 with hybrid weights, w/o softmax
    model = vgg16_hybrid_1365(1)
    # append new softmax layer
    model.add(Dense(train_dataset.nb_classes, activation='softmax'))
    # train with slow SGD
    model.compile(
      optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
      loss='categorical_crossentropy',
      metrics=['accuracy']
    )
    model.fit(X[train_idx],Y[train_idx],
      batch_size = 32,
      # overfits extremely quickly
      epochs = 5,
      validation_data=(X[test_idx],Y[test_idx])
    )
    # evaluate
    score = model.evaluate(X[test_idx],Y[test_idx])
    print(score)
    # only append accuracy score
    scores[i] = score[1]

    i+=1
    K.clear_session()

  print(scores)
  print(np.mean(scores))
  print('done')


def train_and_test_multi(train_datasets):
  # use 3 separate datasets
  img_size = (224,224)
  color_mode = 'grayscale'

  X1,Y1 = train_datasets[0].get_data(img_size,color_mode,1)
  X2,Y2 = train_datasets[1].get_data(img_size,color_mode,1)
  X3,Y3 = train_datasets[2].get_data(img_size,color_mode,1)

  # put into combined numpy array
  X = np.ndarray(shape=(X1.shape[0:3] + (3,)))
  X[:,:,:,0] = X1.squeeze()
  X[:,:,:,1] = X2.squeeze()
  X[:,:,:,2] = X3.squeeze()

  # labels should be the same between datasets (b/c shuffle=false)
  Y = Y1

  labels = np.asarray([np.argmax(y) for y in Y])

  kfold = StratifiedKFold(n_splits=5, shuffle=True)

  scores = [None] * 5
  i=0
  for train_idx, test_idx in kfold.split(X,labels):
    print('fold ' + str(i+1) + ' of 5')

    model = vgg16_hybrid_1365(1)

    model.add(Dense(train_datasets[0].nb_classes, activation='softmax'))
    # train with slow SGD
    model.compile(
      optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
      loss='categorical_crossentropy',
      metrics=['accuracy']
    )
    model.fit(X[train_idx],Y[train_idx],
      batch_size = 32,
      # more epochs since we're using new channels schema
      # approaching RGB performance with e=10...
      # maybe do even more epochs for this method
      #  epochs = 10,
      epochs = 15,
      validation_data=(X[test_idx],Y[test_idx])
    )
    # evaluate
    score = model.evaluate(X[test_idx],Y[test_idx])
    print(score)
    # only append accuracy score
    scores[i] = score[1]

    i+=1
    K.clear_session()
  
  print(scores)
  print(np.mean(scores))
  print('done')



def main():
  # CHOOSE DATASET
  # either a single dataset or a set of 3
  #  dataset_strs = ['rgb']
  dataset_strs = ['line_drawings']
  #  dataset_strs = ['line_drawings', 'dR_symmetric', 'dR_asymmetric']

  datasets = [KFoldDataset(s) for s in dataset_strs]

  train_and_test(datasets)

main()
