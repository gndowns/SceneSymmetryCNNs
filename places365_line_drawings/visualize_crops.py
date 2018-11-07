from keras_utils import crop_generator
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from matplotlib import pyplot as plt

img_size = (256, 256)
color_mode = 'grayscale'
batch_size = 4


train_dir = '/usr/local/data/gabriel/places365_line_drawings/train'
test_dir = '/usr/local/data/gabriel/places365_line_drawings/val'


train_datagen = ImageDataGenerator(
  rescale=1,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
  train_dir,
  target_size = img_size,
  batch_size = batch_size,
  class_mode = 'categorical',
  color_mode = color_mode,
  shuffle=False
)

crop_length = 224


# visualize a single batch (uncropped)
batch_x, batch_y = next(train_gen)
for i in batch_x:
  i = i.squeeze()
  img = Image.fromarray(i)
  img.show()


train_gen = train_datagen.flow_from_directory(
  train_dir,
  target_size = img_size,
  batch_size = batch_size,
  class_mode = 'categorical',
  color_mode = color_mode,
  shuffle=False
)

train_crops = crop_generator(train_gen, crop_length, 1)
# visualize cropped batch
batch_x, batch_y = next(train_crops)
# show each image
for i in batch_x:
  i = i.squeeze()
  img = Image.fromarray(i)
  img.show()
