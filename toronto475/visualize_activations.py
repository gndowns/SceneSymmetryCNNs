# Fine Tune VGG16 on toronto then visualize the activations

from vis.visualization import visualize_activation
from PIL import Image
from k_fold_dataset import KFoldDataset
import numpy as np
from vgg16_utils import vgg16_hybrid_1365, train_top_model, add_top_model

np.random.seed(2018)

def train_and_visualize(dataset):
  # load pre-trained vgg16 hybrid w/o dense layers
  model = vgg16_hybrid_1365(4)

  # standards for VGG16
  img_size = (224, 224)
  color_mode = 'rgb'
  # for places
  rescale = 1


  # load data as numpy arrays
  X,Y = dataset.get_data(img_size, color_mode, rescale)
  
  # generate bottleneck features
  X = model.predict(X)

  # since we're just visualizing activations, we can use the
  # whole dataset for training

  batch_size = 32
  epochs = 50
  lr=1e-5

  top_model_weights = train_top_model(X, Y, X, Y, batch_size, epochs, lr)

  # concatenate base and top model
  base_weights = model.get_weights()
  model = add_top_model(model, dataset.nb_classes)
  # load base & top weights
  model.set_weights(np.concatenate((base_weights, top_model_weights)))

  # visualize actvations for some filters in dense layer
  layer_idx = 19
  # directory to save images to 
  dir_name = 'to_rgb_activations/' if dataset.str == 'rgb' else 'to_ld_activations/'
  for i in range(32):
    print(i)
    activation = visualize_activation(model, layer_idx, i)
    img = Image.fromarray(activation)
    # name format: layer_idx, filter_idx
    img.save(dir_name + 'activation_' + str(layer_idx) + '_' + str(i) + '.png')



def main():
  # choose dataset
  #  dataset_str = 'rgb'
  dataset_str = 'line_drawings'

  dataset = KFoldDataset(dataset_str)

  train_and_visualize(dataset)

main()
