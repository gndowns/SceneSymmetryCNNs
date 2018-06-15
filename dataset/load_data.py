# Helper File for loading different datasets as Keras generators with approrpriate params

from keras.preprocessing.image import ImageDataGenerator

# ========= TORONTO 475 ==========

# Original RGB Images from Toronto475
def toronto_rgb():
  # 6 scene categories: beach,city,forest,highway,mountain,office
  nb_classes = 6
  
  # 76-80 images per category. Split all 75% / 25% for training/testing
  # => ~60 per category for training, ~20 per category for testing
  nb_train_samples = 60 + 59 + 60 + 60 + 57 + 60
  nb_test_samples = 20 + 20 + 20 + 20 + 19 + 20

  # 3 input channels for R-G-B
  nb_channels = 3

  # directories for test and train data, each contains 1 sub-directory per category
  train_dir = 'data/toronto/rgb/train'
  test_dir = 'data/toronto/rgb/test'

  return (nb_classes, nb_train_samples, nb_test_samples, nb_channels, train_dir, test_dir)


# Line Drawings of Toronto 475 rgb images
def toronto_line_drawings():
  # most params are same as toronto_rgb
  nb_classes = 6
  nb_train_samples = 60 + 59 + 60 + 60 + 57 + 60
  nb_test_samples = 20 + 20 + 20 + 20 + 19 + 20
  img_width, img_height = 256,256
  # single input channel for grayscale
  nb_channels = 1

  train_dir = 'data/toronto/line_drawings/train'
  test_dir = 'data/toronto/line_drawings/test'

  return (nb_classes, nb_train_samples, nb_test_samples, nb_channels, train_dir, test_dir)


# 50% nearest contours (minimum R measure on Toronto images)
def to_min_r_near():
  # most params are same as toronto_arc_length_symmetric
  nb_classes = 6
  nb_train_samples = 60 + 59 + 60 + 60 + 57 + 60
  nb_test_samples = 20 + 20 + 20 + 20 + 19 + 20
  # single input channel for grayscale
  nb_channels = 1

  train_dir = 'data/toronto/min_r_near/train'
  test_dir = 'data/toronto/min_r_near/test'

  return (nb_classes, nb_train_samples, nb_test_samples, nb_channels, train_dir, test_dir)

# 50% furthest contours (minimum R measure)
def to_min_r_far():
  # most params are same as toronto_arc_length_symmetric
  nb_classes = 6
  nb_train_samples = 60 + 59 + 60 + 60 + 57 + 60
  nb_test_samples = 20 + 20 + 20 + 20 + 19 + 20
  # single input channel for grayscale
  nb_channels = 1

  train_dir = 'data/toronto/min_r_far/train'
  test_dir = 'data/toronto/min_r_far/test'

  return (nb_classes, nb_train_samples, nb_test_samples, nb_channels, train_dir, test_dir)


# Upper symmetric 50% of toronto line drawings
def toronto_arc_length_symmetric():
  # most params are same as toronto_rgb & toronto_line_drawings
  nb_classes = 6
  nb_train_samples = 60 + 59 + 60 + 60 + 57 + 60
  nb_test_samples = 20 + 20 + 20 + 20 + 19 + 20
  # single input channel for grayscale
  nb_channels = 1

  train_dir = 'data/toronto/arc_length_symmetric/train'
  test_dir = 'data/toronto/arc_length_symmetric/test'

  return (nb_classes, nb_train_samples, nb_test_samples, nb_channels, train_dir, test_dir)


# Lower symmetric 50% of toronto line drawings (arc length normalized)
def toronto_arc_length_asymmetric():
  # most params are same as toronto_arc_length_symmetric
  nb_classes = 6
  nb_train_samples = 60 + 59 + 60 + 60 + 57 + 60
  nb_test_samples = 20 + 20 + 20 + 20 + 19 + 20
  # single input channel for grayscale
  nb_channels = 1

  train_dir = 'data/toronto/arc_length_asymmetric/train'
  test_dir = 'data/toronto/arc_length_asymmetric/test'

  return (nb_classes, nb_train_samples, nb_test_samples, nb_channels, train_dir, test_dir)


# ======== MIT67 =============

# MIT67 Original RGB Images
def mit67_rgb():
  # 67 scene categories
  nb_classes = 67
  
  # These numbers aren't well rounded b/c some image names
  # are repeated in trainImages.txt & testImages.txt
  nb_train_samples = 5354
  nb_test_samples = 1339

  # just use VGG sizes
  img_width, img_height = 256, 256
  
  # rgb, 3 channels
  nb_channels = 3
  
  train_dir = 'data/mit67/rgb/train'
  test_dir = 'data/mit67/rgb/test'
  
  return (nb_classes, nb_train_samples, nb_test_samples, nb_channels, train_dir, test_dir)


# MIT67 Computer Generated Rough Line Drawings (using Dollar Edge Detector)
def mit67_edges():
  # 67 scene categories
  nb_classes = 67
  
  # These numbers aren't well rounded b/c some image names
  # are repeated in trainImages.txt & testImages.txt
  nb_train_samples = 5354
  nb_test_samples = 1339

  # just use VGG sizes
  img_width, img_height = 256, 256
  
  # line drawings are single channle, B/W
  nb_channels = 1
  
  train_dir = 'data/mit67/edges/train'
  test_dir = 'data/mit67/edges/test'
  
  return (nb_classes, nb_train_samples, nb_test_samples, nb_channels, train_dir, test_dir)


# MIT67 Line Drawings (Smoothed from Edges)
def mit67_line_drawings():
  # 67 scene categories
  nb_classes = 67
  
  nb_train_samples = 5354
  nb_test_samples = 1339

  # line drawings are single channle, B/W
  nb_channels = 1

  train_dir = 'data/mit67/line_drawings/train'
  test_dir = 'data/mit67/line_drawings/test'
  
  return (nb_classes, nb_train_samples, nb_test_samples, nb_channels, train_dir, test_dir)

