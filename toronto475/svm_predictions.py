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
  X,Y,train_fnames = train_dataset.get_data(img_size, color_mode,1)

  # init 5-fold cross validation
  kfold = StratifiedKFold(n_splits=5, shuffle=True)

  # one-hot -> class labels
  labels = np.asarray([np.argmax(y) for y in Y])

  # generate bottleneck features (output of conv layers)
  X_bneck = model.predict(X)

  # make array to store filename + prediction
  predictions = [[f, -1] for f in train_fnames] 
  
  i=0
  for train_idx, test_idx in kfold.split(X,labels):
    print('fold ' + str(i+1) + ' of 5')

    # train linear svm
    svc = SVC(kernel='linear')
    svc.fit(X_bneck[train_idx], labels[train_idx])

    # evaluate
    print('predicting...')
    print(train_dataset.str)
    fold_predictions = svc.predict(X_bneck[test_idx]) 

    # add predictions to final list, by index
    for j in range(len(fold_predictions)):
      # get index from test_idx list
      pred_idx = test_idx[j]
      # set prediction value for that specific image file
      predictions[pred_idx][1] = fold_predictions[j]

    i += 1


  # pretty print predictions, one per line
  for p in predictions: print(p)
  print('done')



def main():
  # CHOOSE DATASETS HERE
  # the 1st will be used for training,
  dataset_strs = ['line_drawings']
  #  dataset_strs = ['line_drawings','dR_symmetric', 'dR_asymmetric']
  #  dataset_strs = ['ribbon']
  #  dataset_strs = ['separation']
  #  dataset_strs = ['taper']

  datasets = [KFoldDataset(s) for s in dataset_strs]

  train_and_test(datasets)

main()
