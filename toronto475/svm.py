# Train a linear svm on bottleneck features

from vgg16_utils import vgg16_hybrid_1365
from k_fold_dataset import KFoldDataset
from sklearn.model_selection import StratifiedKFold
from thundersvmScikit import *
import numpy as np

# fix random seed for reproducibility
np.random.seed(2018)

def train_and_test(datasets):
  # use first listed daaset for training
  train_dataset = datasets[0]

  # import vgg16 with hybrid weights, w/o softmax layer
  model = vgg16_hybrid_1365(1)

  # standard
  img_size = (224, 224)
  color_mode = 'rgb'

  # load data as numpy arrays
  # (use rescale=1 for places CNN's)
  X,Y = train_dataset.get_data(img_size, color_mode,1)
  # load testing datasets
  nb_test_sets = len(datasets[1:])
  X_test,Y_test = [None]*nb_test_sets, [None]*nb_test_sets
  for i in range(nb_test_sets):
    X_test[i],Y_test[i] = datasets[i+1].get_data(img_size, color_mode,1)
    # generate bottleneck features
    X_test[i] = model.predict(X_test[i])

  # init 5-fold cross validation
  kfold = StratifiedKFold(n_splits=5, shuffle=True)

  # one-hot -> class labels
  labels = np.asarray([np.argmax(y) for y in Y])

  # generate bottleneck features (output of conv layers)
  X_bneck = model.predict(X)

  # store performance of each fold
  scores = [None] * 5
  test_scores = [None] * 5
  for i in range(nb_test_sets): test_scores[i] = [None] * 5
  i=0
  for train_idx, test_idx in kfold.split(X,labels):
    print('fold ' + str(i+1) + ' of 5')

    # train linear svm
    svc = SVC(kernel='linear')
    svc.fit(X_bneck[train_idx], labels[train_idx])

    # evaluate
    train_score = svc.score(X_bneck[train_idx], labels[train_idx])
    test_score = svc.score(X_bneck[test_idx], labels[test_idx]) 
    print('evaluating...')
    print(train_dataset.str)
    print(train_score, test_score)

    for j in range(nb_test_sets):
      print(datasets[j+1].str)
      score = svc.score(X_test[j][test_idx], labels[test_idx])
      print(score)
      test_scores[j][i] = score


    scores[i] = test_score

    i+=1

  print('\nMeans:')
  print(train_dataset.str)
  print(str(np.mean(scores)))
  for j in range(nb_test_sets):
    print(datasets[j+1].str)
    print(np.mean(test_scores[j]))

  print('done')



def main():
  # CHOOSE DATASETS HERE
  # the 1st will be used for training,
  # all others will be tested on
  #  dataset_strs = ['rgb']
  #  dataset_strs = ['line_drawings','dR_symmetric', 'dR_asymmetric']
  dataset_strs = ['line_drawings', 'dR_weighted']

  datasets = [KFoldDataset(s) for s in dataset_strs]

  train_and_test(datasets)

main()
