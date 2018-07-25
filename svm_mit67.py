# Train a linear SVM classifier on the deep features output by Vgg16_hybrid

from mit67_dataset import MIT67Dataset
from vgg16_utils import places205_vgg16, vgg16_sequential
from keras.applications import VGG16
from vgg16_hybrid_places_1365 import VGG16_Hybrid_1365
from sklearn import preprocessing, svm
from thundersvmScikit import *
import numpy as np
import os.path


def train_and_test(datasets):
  # use first listed dataset for training
  train_dataset = datasets[0]

  # import places205_vgg16 w/o top softmax layer
  #  model = places205_vgg16(1)
  #  vgg  = VGG16()
  #  model = vgg16_sequential(1000)
  #  model.set_weights(vgg.get_weights())
  #  model.pop()
  hybrid = VGG16_Hybrid_1365()
  model = vgg16_sequential(1365)
  model.set_weights(hybrid.get_weights())
  model.pop()


  # check for pre-gen'd features
  # (these should be reset if any params change)
  #  if (os.path.isfile('x_train.npy') and os.path.isfile('y_train.npy')
    #  and os.path.isfile('x_test.npy') and os.path.isfile('y_test.npy')):
    #  print('using saved features...')
    #  bneck_train = np.load('x_train.npy')
    #  bneck_test = np.load('x_test.npy')
    #  train_labels = np.load('y_train.npy')
    #  test_labels = np.load('y_test.npy')

  #  # else generate everyhting now
  #  else: 
  print('generating features...')
  # standard image size for vgg16
  img_size = (224,224)
  # 3 channels for vgg16 compatibility
  color_mode = 'rgb'
  # variable
  batch_size = 16

  # generate numpy arrays of data
  # use rescale=1 to match places conventions
  x_train, y_train = train_dataset.train_data(img_size, color_mode, 1)
  x_test, y_test = train_dataset.test_data(img_size, color_mode, 1)

  # generate bottleneck features (output of conv layers)
  bneck_train = model.predict(x_train,
    batch_size = batch_size,
    verbose=1
  )

  bneck_test = model.predict(x_test,
    batch_size = batch_size,
    verbose=1
  )

  # convert one-hot encodings to class labels
  train_labels = [np.argmax(y) for y in y_train]
  test_labels = [np.argmax(y) for y in y_test]

  # save everything for later
  #  print('saving numpy arrays...')
  #  np.save('x_train.npy', bneck_train)
  #  np.save('x_test.npy', bneck_test)
  #  np.save('y_train.npy', train_labels)
  #  np.save('y_test.npy', test_labels)

  # train svm
  #  m = svm.SVC()
  #  m = svm.SVC(kernel='linear')
  #  m = svm.LinearSVC()

  # Different trials
  print('no preprocessing')
  m = SVC(kernel='linear')
  print('training...')
  m.fit(bneck_train, train_labels)
  print('evaluating...')
  print(m.score(bneck_train, train_labels))
  print(m.score(bneck_test, test_labels))

  print('training on other datasets...')
  for dataset in datasets[1:]:
    print(dataset.str)
    x_test, y_test = dataset.test_data(img_size, color_mode, 1)
    # generate bottleneck features (output of conv layers)
    bneck_test = model.predict(x_test,
      batch_size = batch_size,
      verbose=1
    )
    # convert one-hot encodings to class labels
    test_labels = [np.argmax(y) for y in y_test]
    print(m.score(bneck_test, test_labels))

  print('done')
  
  

def main():
  # choose datasets
  # the first listed will be used for training,
  # all others will be tested on
  #  dataset_strs = ['line_drawings', 'dR_symmetric', 'dR_asymmetric']
  dataset_strs = ['smooth', 'smooth_dR_symmetric', 'smooth_dR_asymmetric']

  datasets = [MIT67Dataset(s) for s in dataset_strs]

  # check for optional preprocessing arg
  train_and_test(datasets)

main()
