# utility methods for Sketch-a-net keras model 
# based on the SketchX paper:
# https://arxiv.org/abs/1501.07873

from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from scipy.io import loadmat
import numpy as np

# Loads sketch-a-net architecture as Sequential model
# with pretrained weights
# trained input size is (224,224), single channel no stroke order
def sketch_a_net():
  model = Sequential([
    # 225x225 is for sketch-a-net compliance, seems they made a mistake in writing 224x224
    Conv2D(64, (15, 15), input_shape=(225,225,1), strides=3, activation='relu'),
    MaxPooling2D(pool_size=(3,3), strides=2),
    Conv2D(128, (5,5), activation='relu'),
    MaxPooling2D(pool_size=(3,3), strides=2),
    Conv2D(256, (3,3), padding='same', activation='relu'),
    Conv2D(256, (3,3), padding='same', activation='relu'),
    Conv2D(256, (3,3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3,3), strides=2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    # Trained on 250 classes (TU-Berlin Sketch dataset)
    Dense(250, activation='softmax')
  ])

  # (load weights)

  # Should be of the same shape as `model.get_weights()`
  # e.g. a list of length 16 
  # each entry should be a numpy array of the appropriate shape:
  # conv2d_1/kernel: (15, 15, 1, 64)
  # conv2d_1/bias:   (64,)
  # conv2d_2/kernel: (5, 5, 64, 128)
  # conv2d_2/bias:   (128,)
  # conv2d_3/kernel: (3, 3, 128, 256)
  # conv2d_3/bias:   (256,)
  # conv2d_4/kernel: (3, 3, 256, 256)
  # conv2d_4/bias:   (256,)
  # conv2d_5/kernel: (3, 3, 256, 256)
  # conv2d_5/bias:   (256,)
  # dense_1/kernel:  (12544, 512)
  # dense_1/bias:    (512,)
  # dense_2/kernel:  (512, 512)
  # dense_2/bias:    (512,)
  # dense_3/kernel:  (512, 250)
  # dense_3/bias:    (250,)

  # load weights from `.mat` file
  weights = sketch_a_net_weights()
  # set weights
  model.set_weights(weights)

  return model

# load the pre-trained weights from the `.mat` file
def sketch_a_net_weights():
  # load .mat file as dict
  model = loadmat('sketch-a-net_without_order_info_224.mat')

  # extract numpy array of layers (21 dimensions)
  # (extra dimensions include relu, pooling layers, etc)
  layers = model['net']['layers'][0,0].squeeze()

  # list that will hold final weights
  weights = [None] * 16

  # indices of relevant conv layers:
  # (others are conv & pooling)
  conv_indices = [0, 3, 6, 8, 10]
  j = 0
  for i in conv_indices:
    # extract single layer (matlab struct)
    layer = layers[i][0,0].squeeze()
    # extract kernel & bias separately
    weights[j] = layer[1]
    j += 1
    weights[j] = layer[2].squeeze()
    j += 1

  # first dense layer
  dense_1 = layers[13][0,0].squeeze()
  # starts as 7x7x256x512 conv kernel
  dense_1_kernel = dense_1[1]
  # reshape to 12544x512 (for dense layer)
  weights[10] = dense_1_kernel.reshape(12544, 512)
  weights[11] = dense_1[2].squeeze()

  # second dense layer
  dense_2 = layers[16][0,0].squeeze()
  # starts as 1x1x512x512, needs only squeezing
  weights[12] = dense_2[1].squeeze()
  weights[13] = dense_2[2].squeeze()

  # final softmax layer
  dense_3 = layers[19][0,0].squeeze()
  # 1x1x512x250, squeeze
  weights[14] = dense_3[1].squeeze()
  weights[15] = dense_3[2].squeeze()

  return weights 
