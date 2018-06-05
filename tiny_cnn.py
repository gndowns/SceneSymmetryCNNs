# helper files for loading datasets and models
from load_data import toronto_rgb, toronto_line_drawings
import load_models
from dataset.dataset import Dataset

from keras.preprocessing.image import ImageDataGenerator

# Global training params
EPOCHS = 10
BATCH_SIZE = 16


def train(model, dataset):
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
    dataset.train_dir,
    # TEMP: fixed
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
  )

  test_gen = test_datagen.flow_from_directory(
    dataset.test_dir,
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
  )

  model.fit_generator(
    train_gen,
    steps_per_epoch = dataset.nb_train_samples // 16,
    epochs = EPOCHS,
    validation_data = test_gen,
    validation_steps = dataset.nb_test_samples // 16
  )



def main():
  # import parameters for chosen dataset
  dataset_str = 'toronto_rgb'
  dataset = Dataset(dataset_str)

  # using fixed iamge size: (256, 256)
  input_shape = (256, 256, dataset.nb_channels)


  # choose model architecture
  models = {
    'mnist': load_models.mnist,
    'cat_dog': load_models.cat_dog
  }
  model_loader = models['cat_dog']

  # load and compile model
  print('loading model architecture...')
  model = model_loader(input_shape, dataset.nb_classes)

  # train model 
  print('training...')
  train(model, dataset)

  # save model
  print('saving model...')
  model.save(dataset_str + '_tiny_cnn.h5')

main()
