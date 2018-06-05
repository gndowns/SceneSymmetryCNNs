# Agnostic testing script-- selects any saved model and dataset and evalutes

from dataset.dataset import Dataset

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Global train/test params
BATCH_SIZE=16

# Evaluate model on test set of given dataset object
def evaluate(model, dataset):
  # grayscale or rgb
  color_mode = 'rgb' if dataset.nb_channels==3 else 'grayscale'

  test_datagen = ImageDataGenerator(rescale=1./255)

  test_gen = test_datagen.flow_from_directory(
    dataset.test_dir,
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    color_mode=color_mode,
    class_mode='categorical'
  )

  score = model.evaluate_generator(
    generator=test_gen,
    steps= dataset.nb_test_samples // BATCH_SIZE
  )

  print(score)


def main():
  # dataset to be used
  dataset_str = 'toronto_line_drawings'
  # h5 file of saved model
  model_file = dataset_str + '_tiny_cnn.h5'

  # load dataset
  dataset = Dataset(dataset_str)
  # load saved model
  model = load_model(model_file)

  evaluate(model, dataset)

main()
