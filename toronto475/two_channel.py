# Use augmented 2 channel images:
# (1) Intact line drawings
# (2) continuous ribbon symmetry weights

from k_fold_dataset import KFoldDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras import backend as K

from mnist import compile_mnist, train_mnist

IMG_SIZE = (224,224)

def train_and_test():
  # load intact line drawings
  intact_dataset = KFoldDataset('toronto_line_drawings')
  X_intact, Y_intact,class_indices = intact_dataset.get_data(IMG_SIZE, 'grayscale')

  # load dR weighted grayscale drawings
  weighted_dataset = KFoldDataset('toronto_dR_weighted')
  X_weighted,Y_weighted,class_indices = weighted_dataset.get_data(IMG_SIZE, 'grayscale')

  # init 5-fold cross validation
  kfold = StratifiedKFold(n_splits=5, shuffle=True)

  # combine datasets into single 2-channel set
  X = np.ndarray(shape=(X_intact.shape[0:3] + (2,)))

  # put each dataset in respective channels
  X[:,:,:,0] = X_intact.squeeze()
  X[:,:,:,1] = X_weighted.squeeze()

  # convert ground-truth one-hot encodings to class labels
  labels = [np.argmax(y) for y in Y_intact]

  two_channel_scores = [None] * 5
  nb_classes = 6
  i=0
  for train_idx, test_idx in kfold.split(X, labels):
    print('fold ' + str(i+1) + ' of 5')

    input_shape = X.shape[1:]
    model = compile_mnist(input_shape, nb_classes)
    model = train_mnist(model,
      X[train_idx], Y_intact[train_idx],
      X[test_idx], Y_intact[test_idx]
    )
    score = model.evaluate(X[test_idx], Y_intact[test_idx])
    two_channel_scores[i] = score[1]

    K.clear_session()
    i+=1

  print(two_channel_scores)
  print(np.mean(two_channel_scores))

train_and_test()
