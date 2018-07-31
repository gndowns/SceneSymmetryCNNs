# Visualize learned filters in pre-trained sketch-a-net
from scipy.misc import imsave
import numpy as np
from sketch_a_net_utils import sketch_a_net

model = sketch_a_net()

# load weights as list
weights = model.get_weights()

# first convolutional layer
layer_idx = 0

# get weights at specific layer
filters = weights[layer_idx]

# visualize all 64 filter
for i in range(64):
  print(i)
  # extract single 15x15 filter
  f = filters[:,:,:,i].squeeze()
  imsave('sketch_a_net_filters/f_' + str(layer_idx) + '_' + str(i) + '.png',
    f)

