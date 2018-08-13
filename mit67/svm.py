# Train a linear SVM classifier on the deep features output by Vgg16_hybrid

from vgg16_utils import vgg16_hybrid_1365
from mit67_dataset import MIT67Dataset
from thundersvmScikit import SVC
import numpy as np

np.random.seed(2018)

def train_and_test(datasets):
  # use first listed dataset for training
  train_dataset = datasets[0]

  model = vgg16_hybrid_1365(1)

  # standards
  img_size = (224,224)
  color_mode = 'rgb'

  # use rescale=1 to match places conventions
  x_train, y_train = train_dataset.train_data(img_size, color_mode, 1)
  x_test, y_test = train_dataset.test_data(img_size, color_mode, 1)

  # generate bottleneck features (output of conv layers)
  bneck_train = model.predict(x_train)
  bneck_test = model.predict(x_test)

  # convert one-hot encodings to class labels
  train_labels = [np.argmax(y) for y in y_train]
  test_labels = [np.argmax(y) for y in y_test]

  # train svm
  svc = SVC(kernel='linear')
  svc.fit(bneck_train, train_labels)

  score = svc.score(bneck_test, test_labels)

  print(score)

  # Different trials
  print('training on other datasets...')
  for dataset in datasets[1:]:
    print(dataset.str)
    x_test, y_test = dataset.test_data(img_size, color_mode, 1)
    # generate bottleneck features (output of conv layers)
    bneck_test = model.predict(x_test)
    # convert one-hot encodings to class labels
    test_labels = [np.argmax(y) for y in y_test]
    print(svc.score(bneck_test, test_labels))

  print('done')
  
  

def main():
  # choose datasets
  # the first listed will be used for training,
  # all others will be tested on
  #  dataset_strs = ['line_drawings', 'dR_symmetric', 'dR_asymmetric']
  dataset_strs = ['smooth', 'smooth_dR_symmetric', 'smooth_dR_asymmetric']
  #  dataset_strs = ['rgb']

  datasets = [MIT67Dataset(s) for s in dataset_strs]

  # check for optional preprocessing arg
  train_and_test(datasets)

main()
