import load_models
from dataset.dataset import Dataset

from keras.preprocessing.image import ImageDataGenerator

import sys

# Global training params
#  EPOCHS = 10
# long term training test
#  EPOCHS = 20
EPOCHS = 100

BATCH_SIZE = 16
# variable
IMG_SIZE = (256, 256)


def train(model, dataset):
  # check if images are grayscale or rgb
  color_mode = 'rgb' if dataset.nb_channels==3 else 'grayscale'

  # use class based generators
  train_gen = dataset.train_gen_aug(color_mode, IMG_SIZE, BATCH_SIZE)
  test_gen = dataset.test_gen(color_mode, IMG_SIZE, BATCH_SIZE)

  model.fit_generator(
    train_gen,
    steps_per_epoch = dataset.nb_train_samples // BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = test_gen,
    validation_steps = dataset.nb_test_samples // BATCH_SIZE
  )


def main():
  # import parameters for chosen dataset
  #  dataset_str = 'toronto_rgb'
  dataset_str = 'toronto_line_drawings'
  #  dataset_str = 'toronto_arc_length_symmetric'
  #  dataset_str = 'toronto_arc_length_asymmetric'
  #  dataset_str = 'to_min_r_near'
  #  dataset_str = 'to_min_r_far'
  #  dataset_str = 'mit67_rgb'
  #  dataset_str = 'mit67_line_drawings'
  #  dataset_str = 'mit67_smooth'




  dataset = Dataset(dataset_str)

  # using fixed image size: (256, 256)
  input_shape = (256, 256, dataset.nb_channels)


  models = {
    'mnist': load_models.mnist,
    'cat_dog': load_models.cat_dog,
    # WIP, ideas taken from sketch-a-net paper
    'sketch_a_net': load_models.sketch_a_net,
    'mnist_large_filters': load_models.mnist_large_filters
  }

  # choose model architecture
  #  model_str = 'cat_dog'
  #  model_str = 'mnist'
  model_str = 'mnist_large_filters'
  #  model_str = 'sketch_a_net'

  model_loader = models[model_str]


  # load and compile model
  print('loading model architecture...')
  model = model_loader(input_shape, dataset.nb_classes)

  # train model 
  print('training...')
  train(model, dataset)

  # save model
  #  print('saving model...')
  #  model.save(dataset_str + '_tiny_cnn.h5')

main()
