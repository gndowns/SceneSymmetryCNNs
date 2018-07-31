# visualize activations of vgg16 pre-trained on places365
from vis.visualization import visualize_activation
from PIL import Image
import numpy as np
from vgg16_utils import vgg16_places365

np.random.seed(2018)

# import pre-trained vgg16
model = vgg16_places365()

# first dense layer
layer_idx = 19

# generate activation for some filters
for i in range(64):
  print(i)
  activation = visualize_activation(model, layer_idx, i)
  img = Image.fromarray(activation)
  # name format: layer_idx, filter_idx
  img.save('vgg16_places365_activations/act_' + str(layer_idx) + '_' + str(i) + '.png')
