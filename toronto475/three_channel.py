# Use augmented 3 channel toronto images:
# (1) intact, (2) symmetric 50%, (3) asymmetric 50%
# the goal is to see if this boosts performance over just the intact
# possible expansion: replace the intact channel with original rgb channel

from k_fold_dataset import KFoldDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras import backend as K

from mnist import compile_mnist, train_mnist

# global training params    
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
# use single channel, since we're going to manually combine the 3 datasets
COLOR_MODE = 'grayscale'
INPUT_SHAPE = IMG_SIZE + (1,)

# fine tunes VGG on the 3 channel images
def train_and_test():
  # load rgb images
  rgb_dataset = KFoldDataset('toronto_rgb')
  X_rgb,Y_rgb,class_indices = rgb_dataset.get_data(IMG_SIZE, 'rgb')

  # load intact line drawings
  intact_dataset = KFoldDataset('toronto_line_drawings')
  X_intact,Y_intact,class_indices = intact_dataset.get_data(IMG_SIZE, COLOR_MODE)

  # load symmetric splits
  sym_dataset = KFoldDataset('toronto_dR_symmetric')
  X_sym,Y_sym,class_indices = sym_dataset.get_data(IMG_SIZE, COLOR_MODE)
  # load asymmetric splits
  asym_dataset = KFoldDataset('toronto_dR_asymmetric')
  X_asym,Y_asym,class_indices = asym_dataset.get_data(IMG_SIZE, COLOR_MODE)

  # init 5-fold cross validation
  kfold = StratifiedKFold(n_splits=5, shuffle=True)

  # combine datasets into single 3-channel set
  # new array to hold combined images
  X = np.ndarray(shape=(X_intact.shape[0:3] + (3,)))

  # put each dataset in respective channels
  X[:,:,:,0] = X_intact.squeeze()
  X[:,:,:,1] = X_sym.squeeze()
  X[:,:,:,2] = X_asym.squeeze()

  # convert ground-truth one-hot encodings to class labels
  labels = [np.argmax(y) for y in Y_intact]

  # train for each fold
  rgb_scores = [None] * 5
  intact_scores = [None] * 5
  three_channel_scores = [None] * 5
  nb_classes = 6
  i=0
  for train_idx, test_idx in kfold.split(X, labels):
    print('fold ' + str(i+1) + ' of 5')

    # train on rgb images ==================
    input_shape = X_rgb.shape[1:]
    model = compile_mnist(input_shape, nb_classes)
    model = train_mnist(model,
      X_rgb[train_idx], Y_rgb[train_idx],
      X_rgb[test_idx], Y_rgb[test_idx]
    )
    score = model.evaluate(X_rgb[test_idx], Y_rgb[test_idx])
    rgb_scores[i] = score[1]

    # Train on just intact images =====
    input_shape = X_intact.shape[1:]
    model = compile_mnist(input_shape, nb_classes)
    model = train_mnist(model,
      X_intact[train_idx], Y_intact[train_idx],
      X_intact[test_idx], Y_intact[test_idx]
    )
    score = model.evaluate(X_intact[test_idx], Y_intact[test_idx])
    intact_scores[i] = score[1]

    # Train on 3-channel combo images ======
    # load model
    input_shape = X.shape[1:]
    model = compile_mnist(input_shape, nb_classes)

    # train model
    model = train_mnist(model,
      X[train_idx], Y_intact[train_idx],
      X[test_idx], Y_intact[test_idx]
    )

    # get final evaluation
    score = model.evaluate(X[test_idx], Y_intact[test_idx])
    # append only accuracy
    three_channel_scores[i] = score[1]

    # clear tensorflow memory
    K.clear_session()

    i+=1

  # print final score
  print(rgb_scores)
  print(np.mean(rgb_scores))
  print(intact_scores)
  print(np.mean(intact_scores))
  print(three_channel_scores)
  print(np.mean(three_channel_scores))

train_and_test()

