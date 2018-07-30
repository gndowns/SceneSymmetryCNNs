# Visualize activations in pre-trained hybrid VGG16
from vis.visualization import visualize_saliency, visualize_activation
from PIL import Image
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
  activation = visualize_activation(model, layer_idx, i)
  img = Image.fromarray(activation)
  # name: layer_idx, filter_idx
  img.save('vgg16_activations/activation_' + str(layer_idx) + '_' + str(i) + '.png')

