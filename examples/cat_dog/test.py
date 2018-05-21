# Test Simple Binary Classifier
# Using images in ./data/test/

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, \
  ImageDataGenerator
import numpy as np

TEST_DATA_PATH = './data/test'

def main():
  # load model
  model = load_model('classifier.h5')

  # generate test data and class labels
  test_datagen = ImageDataGenerator(rescale=1. / 255)
  test_generator = test_datagen.flow_from_directory(
    TEST_DATA_PATH,
    target_size=(150,150),
    batch_size=2,
    class_mode='binary'
  )


  classes = test_generator.class_indices
  # reverse mapping for easy lookup
  classes = {val: key for key,val in classes.iteritems()}


  # load sample image of cat and dog, scale to 150x150
  cat = load_img('./data/test/cat/cat.10901.jpg', target_size=(150,150))
  dog = load_img('./data/test/dog/dog.10901.jpg', target_size=(150,150))
  # convert to numpy array
  cat = img_to_array(cat)
  dog = img_to_array(dog)

  # reshape as batch with 2 samples
  batch = [cat, dog]
  batch = np.asarray(batch)

  # pre-process (scale from 0-255 to 0-1)
  batch = batch / 255.

  p_probs = model.predict(batch)
  p_classes = model.predict_classes(batch)


  # Print results
  print('Predictions: ')
  for p in p_classes:
    print(classes[p[0]])

  print('Ground Truth: ')
  print('cat, dog')
  print('Probabilities: ')
  print(p_probs)



main()
