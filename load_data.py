# Helper File for loading different datasets as Keras generators with approrpriate params

from keras.preprocessing.image import ImageDataGenerator


# Original RGB Images from Toronto475
def toronto_rgb():
  # 6 scene categories: beach,city,forest,highway,mountain,office
  nb_classes = 6
  
  # 76-80 images per category. Split all 75% / 25% for training/testing
  # => ~60 per category for training, ~20 per category for testing
  nb_train_samples = 60 + 59 + 60 + 60 + 57 + 60
  nb_test_samples = 20 + 20 + 20 + 20 + 19 + 20

  # Standard ImageNet / VGG16 image input size
  img_width, img_height = 256, 256
  # 3 input channels for R-G-B
  input_shape = (img_width, img_height, 3)

  # Chosen as highest w/o out of memory failure
  batch_size = 16

  # directories for test and train data, each contains 1 sub-directory per category
  train_dir = 'data/rgb/train'
  test_dir = 'data/rgb/test'

  # Augment training data
  train_datagen = ImageDataGenerator(
    # Rescale 0-255 images to 0-1 range
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
  )

  # For test data, only rescale
  test_datagen = ImageDataGenerator(rescale = 1. / 255)

  # Pull images from class sub-directories
  train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    # one class for each sub-directory/category
    class_mode = 'categorical'
  )
  
  # same for test images
  test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical'
  )

  return (nb_classes, nb_train_samples, nb_test_samples, img_width, \
    img_height, input_shape, batch_size, train_gen, test_gen)

# Line Drawings of Toronto 475 rgb images
def toronto_line_drawings():
  # most params are same as toronto_rgb
  nb_classes = 6
  nb_train_samples = 60 + 59 + 60 + 60 + 57 + 60
  nb_test_samples = 20 + 20 + 20 + 20 + 19 + 20
  img_width, img_height = 256,256
  # single input channel for grayscale
  input_shape = (img_width, img_height, 1)

  batch_size = 16

  train_dir = 'data/line_drawings/train'
  test_dir = 'data/line_drawings/test'

  train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
  )
  test_datagen = ImageDataGenerator(rescale = 1. / 255)

  train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    # uses 1 channel instead of 3 (linedrawings are grayscale)
    color_mode = 'grayscale',
    class_mode = 'categorical'
  ) 

  test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size = (img_widht, img_height),
    batch_size = batch_size,
    color_mode = 'grayscale',
    class_mode = 'categorical'
  )

  return (nb_classes, nb_train_samples, nb_test_samples, img_width, \
    img_height, input_shape, batch_size, train_gen, test_gen)
