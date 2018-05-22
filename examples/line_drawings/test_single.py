# Test current saved model on a single line drawing image

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# path to test image
PATH = './office.png'

def main():
  # laod saved model
  model = load_model('model.h5')

  # load test image as grayscale
  image = load_img(PATH, target_size=(300,300), grayscale=True)
  # convert to numpy array
  image = img_to_array(image)
  # reshape, bactch size 1
  image = image.reshape((1,) + image.shape)
  # rescale 0-255 to 0-1
  image /= 255.

  # get model prediction
  probs = model.predict(image)
  p_class = model.predict_classes(image)

  # generate class labels
  test_datagen = ImageDataGenerator(rescale = 1. / 255)
  test_gen = test_datagen.flow_from_directory(
    'data/test',
    target_size=(300,300),
    batch_size=1,
    class_mode='categorical'
  )

  classes = test_gen.class_indices
  # reverse mapping
  classes = {val: key for key,val in classes.iteritems()}
  print('Classes: ', classes)

  # print
  print('Prediction: ', classes[p_class[0]])
  print('Probabilities: ', probs)

main()
