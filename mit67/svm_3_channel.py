# Train a linear SVM on three channel line drawing images:
# the 3 channels can be any selected Toronto datasets
# e.g. Intact + Symmetric + Asymmetric
# since they're all part of Toronto475, the labels should be the same for each

from vgg16_utils import vgg16_hybrid_1365
from mit67_dataset import MIT67Dataset
from thundersvmScikit import SVC
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
  x1_train, y1_train = datasets[0].train_data(img_size, color_mode,1)
  x2_train, y2_train = datasets[1].train_data(img_size, color_mode,1)
  x3_train, x3_test = datasets[2].train_data(img_size, color_mode,1)

  x1_test, y1_test = datasets[0].test_data(img_size, color_mode, 1)
  x2_test, y2_test = datasets[1].test_data(img_size, color_mode, 1)
  x3_test, y3_test = datasets[2].test_data(img_size, color_mode, 1)

  # combined numpy array to hold data together
  x_train = np.ndarray(shape=(x1_train.shape[0:3] + (3,)))
  x_test = np.ndarray(shape=(x1_test.shape[0:3] + (3,)))
  # put each dataset in respective channels
  x_train[:,:,:,0] = x1_train.squeeze()
  x_train[:,:,:,1] = x2_train.squeeze()
  x_train[:,:,:,2] = x3_train.squeeze()
  x_test[:,:,:,0] = x1_test.squeeze()
  x_test[:,:,:,1] = x2_test.squeeze()
  x_test[:,:,:,2] = x3_test.squeeze()

  # one-hot -> class labels
  train_labels = np.asarray([np.argmax(y) for y in y1_train])
  test_labels = np.asarray([np.argmax(y) for y in y1_test])

  # generate bottleneck features (output of conv layers)
  bneck_train = model.predict(x_train)
  bneck_test = model.predict(x_test)

  # train linear svm
  svc = SVC(kernel='linear')
  print('training svm...')
  svc.fit(bneck_train, train_labels)

  # evaluate
  score = svc.score(bneck_test, test_labels) 
  print(score)

  print('done')


def main():
  # CHOOSE 3 datasets
  #  dataset_strs = ['smooth', 'ribbon', 'taper']
  #  dataset_strs = ['smooth', 'smooth', 'smooth']
  #  dataset_strs = ['smooth', 'ribbon', 'ribbon']
  #  dataset_strs = ['smooth', 'taper', 'taper']
  #  dataset_strs = ['smooth', 'smooth', 'ribbon']
  #  dataset_strs = ['ribbon', 'ribbon', 'smooth']
  #  dataset_strs = ['smooth', 'separation', 'separation']
  #  dataset_strs = ['smooth', 'ribbon', 'separation']
  #  dataset_strs = ['smooth', 'taper', 'separation']
  dataset_strs = ['ribbon', 'taper', 'separation']



  print('using datasets: ')
  print(dataset_strs)

  datasets = [MIT67Dataset(s) for s in dataset_strs]

  train_and_test(datasets)

main()
