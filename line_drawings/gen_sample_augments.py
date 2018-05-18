# Applies some sample transforms to one beach image, puts results in 'preview/' directory
# Code taken from: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def augment():
  # Generate Augmented Training Data
  train_datagen = ImageDataGenerator(
    # scale 0--255 pixels to 0-1
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
  )

  # demonstrate rescaling with single image
  img = load_img('data/arc_length/intact/train/beach/Beach_1_0_18.png', target_size=(224,224))
  # convert to numpy array
  img = img_to_array(img)
  # add extra channel (for batch size)
  img = img.reshape((1,) + img.shape)

  # generate 20 different re-scaled images, datagen applied to img
  i = 0
  for batch in train_datagen.flow(img, batch_size=1,
    save_to_dir='preview', save_prefix='beach', save_format='jpeg'):
    i += 1
    if i > 20:
      break


augment()
