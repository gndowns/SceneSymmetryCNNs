# Visualize activations in pre-trained hybrid VGG16
from vis.visualization import visualize_activation
from scipy.misc import imsave
import numpy as np
from vgg16_utils import vgg16_hybrid_1365

np.random.seed(2018)

# import pre-trained vgg16
model = vgg16_hybrid_1365()

# first dense layer
layer_idx = 19

# generate activations for all filters
#  activation = visualize_activation(model, layer_idx)

# generate activations for some filters at specified layer
for i in range(32):
  print(i)
  activation = visualize_activation(model, layer_idx, i)
  # name: layer_idx, filter_idx
  imsave(('vgg16_hybrid_1365_activations/' +
    'act_' + str(layer_idx) + '_' + str(i) + '.png'),
    activation)

