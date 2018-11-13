# Train a linear SVM on three channel line drawing images:
# the 3 channels can be any selected Toronto datasets
# e.g. Intact + Symmetric + Asymmetric
# since they're all part of Toronto475, the labels should be the same for each

from vgg16_utils import vgg16_hybrid_1365
from k_fold_dataset import KFoldDataset
from sklearn.model_selection import StratifiedKFold
from thundersvmScikit import *
import numpy as np

# fix random seed for reproducibility
np.random.seed(2018)

def train_and_test(datasets):
  # import vgg16 with hybrid weights, w/o softmax layer
  model = vgg16_hybrid_1365(1)

  # standard
  img_size = (224, 224)
  color_mode = 'grayscale'

  # load 3 datasets separately
  X1,Y1 = datasets[0].get_data(img_size, color_mode,1)
  X2,Y2 = datasets[1].get_data(img_size, color_mode,1)
  X3,Y3 = datasets[2].get_data(img_size, color_mode,1)

  # combined numpy array to hold data together
  X = np.ndarray(shape=(X1.shape[0:3] + (3,)))
  # put each dataset in respective channels
  X[:,:,:,0] = X1.squeeze()
  X[:,:,:,1] = X2.squeeze()
  X[:,:,:,2] = X3.squeeze()

  # init 5-fold cross validation
  kfold = StratifiedKFold(n_splits=5, shuffle=True)

  # one-hot -> class labels
  labels = np.asarray([np.argmax(y) for y in Y1])

  # generate bottleneck features (output of conv layers)
  X_bneck = model.predict(X)

  # store performance of each fold
  scores = [None] * 5
  i=0
  for train_idx, test_idx in kfold.split(X,labels):
    print('fold ' + str(i+1) + ' of 5')

    # train linear svm
    svc = SVC(kernel='linear')
    svc.fit(X_bneck[train_idx], labels[train_idx])

    # evaluate
    train_score = svc.score(X_bneck[train_idx], labels[train_idx])
    test_score = svc.score(X_bneck[test_idx], labels[test_idx]) 
    print(train_score, test_score)

    scores[i] = test_score

    i+=1

  print('mean: ' + str(np.mean(scores)))
  print('done')


def main():
  # CHOOSE 3 datasets
  #  dataset_strs = ['line_drawings', 'dR_symmetric', 'dR_asymmetric']
  #  dataset_strs = ['line_drawings', 'dR_weighted', 'd2R_weighted']
  #  dataset_strs = ['line_drawings', 'dR_weighted', 'dR_weighted']
  #  dataset_strs = ['line_drawings', 'line_drawings', 'dR_weighted']
  #  dataset_strs = ['line_drawings', 'd2R_weighted', 'd2R_weighted']
  #  dataset_strs = ['line_drawings', 'line_drawings', 'd2R_weighted']
  #  dataset_strs = ['line_drawings', 'd2R_weighted', 'dR_weighted']
  #  dataset_strs = ['dR_weighted', 'line_drawings', 'd2R_weighted']
  #  dataset_strs = ['dR_weighted', 'd2R_weighted', 'line_drawings']
  #  dataset_strs = ['d2R_weighted', 'line_drawings', 'dR_weighted']
  #  dataset_strs = ['d2R_weighted', 'dR_weighted', 'line_drawings']

  # controls
  #  dataset_strs = ['line_drawings', 'max_R', 'max_R']
  #  dataset_strs = ['line_drawings', 'min_R', 'min_R']
  #  dataset_strs = ['line_drawings', 'min_R', 'max_R']
  #  dataset_strs = ['line_drawings', 'dollar_weighted', 'dollar_weighted']

  #  dataset_strs = ['line_drawings', 'ribbon', 'separation']
  dataset_strs = ['line_drawings', 'taper', 'separation']


  datasets = [KFoldDataset(s) for s in dataset_strs]

  train_and_test(datasets)

main()
