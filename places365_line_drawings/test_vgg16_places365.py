# Test pre-trained places365 vgg16 network on feature-specific places datasets

from vgg16_utils import vgg16_places365
from places365_dataset import Places365Dataset
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np

# seed for reproducibility
np.random.seed(2018)

# constants, for compliance w/ pre-trained places vgg
IMG_SIZE = (224, 224)
COLOR_MODE = 'rgb'
RESCALE = 1
# machine specific-- anything higher causes oom error
BATCH_SIZE = 64


def train_and_test(datasets):
  # use first dataset for training svm
  train_dataset = datasets[0]

  # load pre-trained model as is
  model = vgg16_places365()
 
  # load train/test data with no data augmentation
  datagen = ImageDataGenerator(rescale=RESCALE)

  test_gen = datagen.flow_from_directory(
    train_dataset.test_dir,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    color_mode = COLOR_MODE
  )

  # compile model
  model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=1e-3, momentum=0.9),
    metrics=['accuracy']
  )

  # just run pre-trained network on test set of new dataset
  print('evaluating...')
  score = model.evaluate_generator(
    test_gen,
    steps = train_dataset.nb_test_samples / BATCH_SIZE
  )

  print(score)


def main():
  # choose dataset
  dataset_strs = ['line_drawings']

  datasets = [Places365Dataset(s) for s in dataset_strs]

  train_and_test(datasets)

main()
