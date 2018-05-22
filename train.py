# helper files for loading datasets and models
from load_data import toronto_rgb, toronto_line_drawings

def main():
  # Choose and load dataset from options
  datasets = {
    'toronto_rgb': toronto_rgb,
    'toronto_line_drawings': toronto_line_drawings
  }
  dataset = datasets['toronto_rgb']

  # use dictionary to call helper method to import data
  nb_classes, nb_train_samples, nb_test_samples, img_width, img_height, \
    input_shape, batch_size, train_gen, test_gen = dataset()

main()
