# helper files for loading datasets and models
from load_data import toronto_rgb, toronto_line_drawings
from load_models import mnist, cat_dog

from keras.preprocessing.image import ImageDataGenerator


def train(model, train_dir, test_dir, nb_train_samples, nb_test_samples):
  # Generator for Data augmentation
  train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
  )
  # test generator
  test_datagen = ImageDataGenerator(rescale=1./255)

  train_gen = train_datagen.flow_from_directory(
    train_dir,
    # TEMP: fixed
    target_size=(256, 256),
    batch_size=16,
    class_mode='categorical'
  )

  test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=16,
    class_mode='categorical'
  )

  model.fit_generator(
    train_gen,
    steps_per_epoch = nb_train_samples // 16,
    epochs = 10,
    validation_data = test_gen,
    validation_steps = nb_test_samples // 16
  )



def main():
  # Choose and load dataset from options
  datasets = {
    'toronto_rgb': toronto_rgb,
    'toronto_line_drawings': toronto_line_drawings
  }
  dataset = datasets['toronto_rgb']

  # import parameters for chosen dataset
  nb_classes, nb_train_samples, nb_test_samples, img_width, img_height, \
    input_shape, batch_size, train_dir, test_dir, train_gen, test_gen = dataset()

  # choose model architecture
  models = {
    'mnist': mnist,
    'cat_dog': cat_dog
  }
  model = models['cat_dog']

  # laod and compile model
  model = model(input_shape, nb_classes)

  # train model 
  train(model, train_dir, test_dir, nb_train_samples, nb_test_samples)

main()
