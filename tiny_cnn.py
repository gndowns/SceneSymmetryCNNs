# helper files for loading datasets and models
from load_data import toronto_rgb, toronto_line_drawings
from load_models import mnist, cat_dog

def main():
  # Choose and load dataset from options
  datasets = {
    'toronto_rgb': toronto_rgb,
    'toronto_line_drawings': toronto_line_drawings
  }
  dataset = datasets['toronto_rgb']

  # import parameters for chosen dataset
  nb_classes, nb_train_samples, nb_test_samples, img_width, img_height, \
    input_shape, batch_size, train_gen, test_gen = dataset()

  # choose model architecture
  models = {
    'mnist': mnist,
    'cat_dog': cat_dog
  }
  model = models['cat_dog']

  # laod and compile model
  model = model(input_shape, nb_classes)


main()
