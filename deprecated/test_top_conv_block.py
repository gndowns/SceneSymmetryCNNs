# Evaluate Saved Model after fine tuning Top Conv Block

import load_data
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Convert single channel B/W images to 3 channels (all equal)
def to_rgb(input_shape, train_dir, test_dir, img_width, img_height, batch_size):
  # augment to 3 channels
  input_shape = (input_shape[0], input_shape[1], 3)

  # data generators
  train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
  )
  test_datagen = ImageDataGenerator(rescale = 1. / 255)
  
  # generators, leave out 'grayscale' color_mode option
  train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical'
  )
  test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical'
  )

  return (input_shape, train_gen, test_gen)

def get_model(dataset_str):
  # first load full model for architecture
  model = load_model(dataset_str + '_top_conv_block.h5')

  # then load best performing weights
  model.load_weights(dataset_str + '_top_conv_block_weights.h5')

  return model
  

# evaluate model on test generator
def evaluate(model, test_gen, steps):
  score = model.evaluate_generator(
    generator=test_gen,
    steps=steps
  )
  print(score)
  

def main():
  # links to data loader functions
  datasets = {
    'toronto_rgb': load_data.toronto_rgb,
    'toronto_line_drawings': load_data.toronto_line_drawings,
    'mit67_rgb': load_data.mit67_rgb,
    'mit67_edges': load_data.mit67_edges,
    'mit67_line_drawings': load_data.mit67_line_drawings
  }
  # SELECT DATASET HERE
  dataset_str = 'mit67_line_drawings'

  dataset = datasets[dataset_str]

  print('Using dataset ' + dataset_str)

  # import data
  nb_classes, nb_train_samples, nb_test_samples, img_width, img_height, \
    input_shape, batch_size, train_dir, test_dir, train_gen, test_gen = dataset()
  
  # for black/white datasets, need to convert to 3 channels for VGG
  if input_shape[2] == 1:
    input_shape, train_gen, test_gen = to_rgb(input_shape, train_dir, test_dir,
      img_width, img_height, batch_size)

  print('loading saved model and weights...')
  model = get_model(dataset_str)

  print('evaluating model...')
  evaluate(model, test_gen, nb_test_samples // batch_size)

main()
