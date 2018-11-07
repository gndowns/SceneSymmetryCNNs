# Train new softmax classifier using 224 cropping method
from mit67_dataset import MIT67Dataset
from vgg16_utils import vgg16_hybrid_1365
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
from keras_utils import crop_generator
from keras.preprocessing.image import ImageDataGenerator

def train_and_test(datasets):
  train_dataset = datasets[0]

  model = vgg16_hybrid_1365(1)
  model.add(Dense(67, activation='softmax'))
  model.compile(
    optimizer=SGD(lr=1e-3,decay=1e-6,momentum=0.9,nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  img_size = (256, 256)
  color_mode = 'rgb'
  batch_size = 64

  train_datagen = ImageDataGenerator(
    rescale=1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
  )

  test_datagen = ImageDataGenerator(rescale=1)

  train_gen = train_datagen.flow_from_directory(
    train_dataset.dir + '/train/',
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    color_mode = color_mode
  )

  test_gen = test_datagen.flow_from_directory(
    train_dataset.dir + '/test/',
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    color_mode = color_mode
  )

  crop_length = 224
  train_crops = crop_generator(train_gen, crop_length, 3)
  test_crops = crop_generator(test_gen, crop_length, 3)

  model.fit_generator(
    train_crops,
    steps_per_epoch = train_dataset.nb_train_samples / batch_size,
    validation_data = test_crops,
    validation_steps = train_dataset.nb_test_samples / batch_size,
    epochs = 10
  )
    
def main():
  dataset_str = 'smooth'
  datasets = [MIT67Dataset(dataset_str)]

  train_and_test(datasets)

main()
