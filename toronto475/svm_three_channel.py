# Train a linear SVM on three channel line drawing images:
# intact + symmetric + asymmetric

from vgg16_utils import vgg16_hybrid_1365
from k_fold_dataset import KFoldDataset
from sklearn.model_selection import StratifiedKFold
from thundersvmScikit import *
import numpy as np

# fix random seed for reproducibility
np.random.seed(2018)

def train_and_test():
  # import vgg16 with hybrid weights, w/o softmax layer
  model = vgg16_hybrid_1365(1)

  # standard
  img_size = (224, 224)
  color_mode = 'grayscale'

  # load intact line drawings
  intact_dataset = KFoldDataset('line_drawings')
  X_intact,Y_intact,class_indices = intact_dataset.get_data(img_size, color_mode)

  # load symmetric splits
  sym_dataset = KFoldDataset('dR_symmetric')
  X_sym,Y_sym,class_indices = sym_dataset.get_data(img_size, color_mode)
  # load asymmetric splits
  asym_dataset = KFoldDataset('dR_asymmetric')
  X_asym,Y_asym,class_indices = asym_dataset.get_data(img_size, color_mode)

  # combined numpy array to hold data together
  X = np.ndarray(shape=(X_intact.shape[0:3] + (3,)))
  # put each dataset in respective channels
  X[:,:,:,0] = X_intact.squeeze()
  X[:,:,:,1] = X_sym.squeeze()
  X[:,:,:,2] = X_asym.squeeze()

  # init 5-fold cross validation
  kfold = StratifiedKFold(n_splits=5, shuffle=True)

  # one-hot -> class labels
  labels = np.asarray([np.argmax(y) for y in Y_intact])

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


train_and_test()
