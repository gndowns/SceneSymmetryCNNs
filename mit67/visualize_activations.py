# Fine Tune Vgg16_hybrid on MIT67 and then visualize activations

from vis.visualization import visualize_activation
from keras.models import load_model
from scipy.misc import imsave
import numpy as np

np.random.seed(2018)

def main():
  #  dataset_str = 'rgb'
  dataset_str = 'smooth'

  # load fine tuned model
  #  model = train_and_test([dataset])
  model = load_model('models/vgg16_hybrid_1365_softmax_mit67_' + dataset_str + '.h5')
  
  # use first dense layer
  #  layer_idx = 19
  # first conv layer
  #  layer_idx = 0

  # visualize all convolutional & fully connected layers
  layer_indices = [0,1,3,4,6,7,8,10,11,12,14,15,16,19,20,21]

  nb_filters = 32

  # for each layer, iterate over nb_filters many filters
  for layer_idx in layer_indices:
    print('layer: ' + str(layer_idx))
    for filter_idx in range(nb_filters):
      print('filter: ' + str(filter_idx))
      act = visualize_activation(model, layer_idx, filter_idx).squeeze()

      # directory to save images to
      dir_name = 'mit67_' + dataset_str + '_activations/'

      imsave((dir_name + 'act_' + str(layer_idx) +
        '_' + str(filter_idx) + '.png'),
        act
      )

main()
