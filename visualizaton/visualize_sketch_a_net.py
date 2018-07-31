# Visualize activations of pre-trained sketch-a-net
from vis.visualization import visualize_activation
from PIL import Image
import numpy as np
from sketch_a_net_utils import sketch_a_net

np.random.seed(2018)

# pre-trained sketch-a-net
model = sketch_a_net()

# first dense layer
#  layer_idx = 9
# second dense layer
#  layer_idx = 11
# first conv layer
layer_idx = 0

# visualize first 64 filters
for i in range(64):
  print(i)
  activation = visualize_activation(model, layer_idx, i)
  # convert to 3-channel rgb image
  img = np.stack((activation,activation,activation), axis=2).squeeze()
  img = Image.fromarray(img)
  # naming: layer_idx, filter_idx
  img.save('sketch_a_net_activations/act_' + str(layer_idx) + '_' + str(i) + '.png')
