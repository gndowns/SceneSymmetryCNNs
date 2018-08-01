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
  layer_idx = 0
  nb_filters = 64
  
  # generate activations for first 32 neurons
  for i in range(nb_filters):
    print(i)
    act = visualize_activation(model, layer_idx, i).squeeze()

    # directory to save images to
    dir_name = 'mit67_' + dataset_str + '_activations/'

    imsave(dir_name + 'act_' + str(layer_idx) + '_' + str(i) + '.png', 
      act)

main()
