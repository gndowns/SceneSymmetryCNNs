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

# fixed seed for reproducibility
seed = 2018
np.random.seed(seed)

# global training params
EPOCHS = 10
BATCH_SIZE = 16
IMG_SIZE = (224,224)
COLOR_MODE = 'rgb'
# 3 channels for VGG compatability 
INPUT_SHAPE = IMG_SIZE + (3,)


# train and test with k-fold cross validation
def train_and_test(datasets):
  train_dataset = datasets[0]

  # load data for training the model
  X,Y,class_indices = train_dataset.get_data(IMG_SIZE, COLOR_MODE)

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
  # CHOOSE DATASETS HERE
  # this should be a list of strings of the form
  # ['train_dataset', 'test_dataset_1', 'test_dataset_2', ...]
  # A model will be trained on the first dataset
  # (`train_dataset` here),
  # the trained model is then tested on each included `test_dataset`
  # the model will by default always be tested on the `train_dataset`
  dataset_strs = ['line_drawings']

  # convert from strings to KFoldDataset objects
  datasets= [KFoldDataset(s) for s in dataset_strs]

  train_and_test(datasets)


main()
