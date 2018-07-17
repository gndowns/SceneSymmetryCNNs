# Fine tune VGG16 using k-fold cross validation
# First tunes top dense layers, then 
# top layers with convolutional layers all together


from k_fold_dataset import KFoldDataset
#  import load_models
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
import numpy as np

from top_model import bottleneck_features, train_top_model
from top_conv_block import train_top_conv_block

# global training params
EPOCHS = 10
BATCH_SIZE = 16
IMG_SIZE = (224,224)
COLOR_MODE = 'rgb'
# 3 channels for VGG compatability 
INPUT_SHAPE = IMG_SIZE + (3,)


# implement k-fold cross validation
def k_fold_cross_val(dataset):
  # load data
  X,Y,class_indices = dataset.get_data(IMG_SIZE, COLOR_MODE)

  # init 5-fold cross validation
  kfold = StratifiedKFold(n_splits=5, shuffle=True)

  # get class labels from one-hot encodings (required for sklearn kfold method)
  labels = [np.argmax(y) for y in Y]

  # generate bottleneck features (output of conv layers of VGG16)
  X_bottleneck = bottleneck_features(X) 


  scores = []
  i=1
  # train & test a model for each fold
  for train_idx, test_idx in kfold.split(X,labels):
    print('fold ' + str(i) + ' of 5')

    # train top model on bottleneck features
    top_model = train_top_model(
      X_bottleneck[train_idx], Y[train_idx],
      X_bottleneck[test_idx], Y[test_idx]
    )

    # tune top layers of VGG model, 
    # using trained top model weights
    model = train_top_conv_block(
      X[train_idx], Y[train_idx],
      X[test_idx], Y[test_idx],
      top_model
    )

    # evaluate trained model
    score = model.evaluate(X[test_idx], Y[test_idx])

    # append accuracy score only
    scores.append(score[1])

    # clear tensorflow memory
    K.clear_session()

    i += 1

  print(scores)
  print(np.mean(scores))



def main():
  # CHOOSE DATASET HERE
  dataset_str = 'toronto_line_drawings'

  # init dataset object
  dataset = KFoldDataset(dataset_str)

  k_fold_cross_val(dataset)

main()
