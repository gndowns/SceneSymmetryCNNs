# Visualize Saliency of VGG16 Hybrid fine tuned on MIT67

from vis.visualization import visualize_saliency
from keras.models import load_model
from scipy.misc import imsave
from mit67_dataset import MIT67Dataset


def main():
  #  dataset_str = 'rgb'
  dataset_str = 'smooth'

  dataset = MIT67Dataset(dataset_str)

  model = load_model('models/vgg16_hybrid_1365_softmax_mit67_' + dataset_str + '.h5')

  # first dense layer
  layer_idx = 19

  # choose image (indexed by unshuffled test set)
  img_idx = 0
  X,Y = dataset.test_data((224,224), 'rgb', 1)

  img = X[img_idx]

  # directory to save images to
  dir_name = 'mit67_' + dataset_str + '_saliency/'
  # save unaltered image for comparison
  imsave(dir_name + 'smooth_' + str(img_idx) + '.png', img)

  # try seveal filters at this layer
  nb_filters = 64

  for i in range(nb_filters):
    print(i)
    saliency = visualize_saliency(model, layer_idx, i, img)

    # name format: img_index+layer_index+filter_index
    imsave((dir_name + 'sal_' + str(img_idx) + 
      '_' + str(layer_idx) + '_' + str(i) + '.png'),
      saliency
    )


main()
