# Replace softmax and retrain whole network

'''
still doesn't work with TF GPU
# reproducibility, taken from:
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

# must all be set before any imports 
import numpy as np
import tensorflow as tf
import random as rn

np.random.seed(2018)

# something to do with python random numbers
rn.seed(2019)

# force Tensorflow to use single thread for reproducibility
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K
# Tensorflow has its own random number generator independ of numpy,
# which must also be seeded
tf.set_random_seed(2020)

# Tensorflow initial state
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
'''


from mit67_dataset import MIT67Dataset
from vgg16_utils import vgg16_hybrid_1365, vgg16_hybrid_1365_stride
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K
import numpy as np

def train_and_test(datasets, model_str):
  train_dataset = datasets[0]

  img_size = (224,224)
  color_mode = 'rgb'

  x_train, y_train = train_dataset.train_data(img_size,color_mode,1)
  x_test, y_test = train_dataset.test_data(img_size,color_mode,1)
  
  # load testing datasets
  nb_test_sets = len(datasets[1:])
  xs_test, ys_test = [None]*nb_test_sets, [None]*nb_test_sets
  for j,d in enumerate(datasets[1:]):
    xs_test[j],ys_test[j] = d.test_data(img_size,color_mode,1)

  # repeat 5 times and take the mean
  # since results are not perfectly reproducible
  train_scores = [None] * 5
  test_scores = [[None]*5 for _ in range(nb_test_sets)]
  for i in range(5):
    print('trial ' + str(i+1) + ' of 5')
    # load proper model 
    if model_str == 'vgg16_hybrid_1365':
      model = vgg16_hybrid_1365(1)
    elif model_str == 'vgg16_hybrid_1365_stride':
      model = vgg16_hybrid_1365_stride(1)
    else:
      print 'ERROR: model not implemented'
      return

    # append new softmax layer
    model.add(Dense(train_dataset.nb_classes, 
      # for reproducibility
      #  kernel_initializer=glorot_uniform(seed=2018),
      activation='softmax'))

    # fine tuning stats
    model.compile(
      optimizer=SGD(lr=1e-3,decay=1e-6,momentum=0.9,nesterov=True),
      loss='categorical_crossentropy',
      metrics=['accuracy']
    )

    model.fit(x_train, y_train,
      batch_size = 32,
      # converges very quickly, but need more (10) for mit67
      #  epochs = 5,
      epochs = 10,
      validation_data = (x_test, y_test)
    )

    # take accuracy score
    train_scores[i] = model.evaluate(x_test,y_test)[1]

    print('testing on other datasets...')
    for j,d in enumerate(datasets[1:]):
      print(d.str)
      test_scores[j][i] = model.evaluate(xs_test[j],ys_test[j])[1]

    # clear memory
    K.clear_session()

  # mean results
  print(train_dataset.str)
  print(train_scores)
  print(np.mean(train_scores))
  for j,d in enumerate(datasets[1:]):
    print(d.str)
    print(test_scores[j])
    print(np.mean(test_scores[j]))

  print('done')



def main():
  #  dataset_strs = ['rgb']
  #  dataset_strs = ['smooth']
  dataset_strs = ['smooth', 'smooth_dR_symmetric', 'smooth_dR_asymmetric']

  datasets = [MIT67Dataset(s) for s in dataset_strs]

  # choose which vgg16 model to use
  #  model_str = 'vgg16_hybrid_1365'
  model_str = 'vgg16_hybrid_1365_stride'

  train_and_test(datasets, model_str)

main()
