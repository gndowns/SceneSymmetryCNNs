# Visualize activations of pre-trained sketch-a-net
from vis.visualization import visualize_activation
from scipy.misc import imsave
import numpy as np
from sketch_a_net_utils import sketch_a_net

np.random.seed(2018)

# pre-trained sketch-a-net
model = sketch_a_net()

#  first dense layer
#  layer_idx = 9
# second dense layer
#  layer_idx = 11
#  first conv layer
layer_idx = 0
nb_filters = 64

# visualize first 64 filters
for i in range(nb_filters):
  print(i)
  activation = visualize_activation(model, layer_idx, i).squeeze()
  imsave('sketch_a_net_activations/act_' + str(layer_idx) + '_' + str(i) + '.png',
    activation)
