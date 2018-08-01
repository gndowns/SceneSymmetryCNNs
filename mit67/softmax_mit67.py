# Only re-train softmax layer of model
# Then use re-trained network for an SVM

from mit67_dataset import MIT67Dataset
from vgg16_utils import vgg16_hybrid_1365
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

np.random.seed(2018)

def train_and_test(train_dataset):
  # load vgg16 w/o top layer
  model = vgg16_hybrid_1365(1)

  # replace with our own softmax layer
  model.add(Dense(train_dataset.nb_classes, activation='softmax'))

  # freeze everything but top conv block & dense layers
  #  for l in model.layers[:-8]:
    #  l.trainable = False

  # standard image size for vgg16
  img_size = (224,224)
  # 3 channels for vgg16 compatibility
  color_mode = 'rgb'
  # generate numpy arrays of all data
  # use rescale=1 to match places conventions
  x_train, y_train = train_dataset.train_data(img_size, color_mode, 1)
  x_test, y_test = train_dataset.test_data(img_size, color_mode, 1)

  # train with low learning rate
  model.compile(
    optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  model.fit(x_train, y_train,
    # Each epoch takes about a minute, converges very fast
    epochs = 5,
    batch_size = 32,
    validation_data = (x_test, y_test)
  )

  return model



def main():
  #  dataset_str = 'rgb'
  dataset_str = 'smooth'
  dataset = MIT67Dataset(dataset_str)
  model = train_and_test(dataset)

  model.save('models/vgg16_hybrid_1365_softmax_mit67_' + dataset_str + '.h5')


main()
