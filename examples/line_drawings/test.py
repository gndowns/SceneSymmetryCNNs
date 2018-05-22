# Evaluate model on Test Data

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

TEST_DATA_DIR = 'data/arc_length/intact/test'

NUM_SAMPLES = 6 * 20
BATCH_SIZE = 16

def main():
  # load previously trained model
  model = load_model('model.h5')

  test_datagen = ImageDataGenerator(rescale=1. / 255)
  # generate all images & labels from directories
  test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size= (300, 300),
    batch_size= BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical'
  )

  score = model.evaluate_generator(
    generator = test_generator,
    steps = NUM_SAMPLES // BATCH_SIZE
  )

  print(score)

main()
