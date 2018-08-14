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
from vgg16_utils import vgg16_hybrid_1365
from keras.layers import Dense
from keras.optimizers import SGD
#  from keras.initializers import glorot_uniform

def train_and_test(datasets):
  train_dataset = datasets[0]

  model = vgg16_hybrid_1365(1)

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

  img_size = (224,224)
  color_mode = 'rgb'

  x_train, y_train = train_dataset.train_data(img_size,color_mode,1)
  x_test, y_test = train_dataset.test_data(img_size,color_mode,1)


  model.fit(x_train, y_train,
    batch_size = 32,
    # converges very quickly
    epochs = 5,
    validation_data = (x_test, y_test)
  )

  score = model.evaluate(x_test,y_test)
  print(score)

  print('training on other datasets...')
  for d in datasets[1:]:
    print(d.str)
    x_test,y_test = d.test_data(img_size,color_mode,1)
    score = model.evaluate(x_test,y_test)
    print(score)

  print('done')



def main():
  #  dataset_strs = ['rgb']
  dataset_strs = ['smooth']
  #  dataset_strs = ['smooth', 'smooth_dR_symmetric', 'smooth_dR_asymmetric']

  datasets = [MIT67Dataset(s) for s in dataset_strs]

  train_and_test(datasets)

main()
